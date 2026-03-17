use std::{collections::HashMap, path::Path, process::Stdio, sync::Arc};

use async_trait::async_trait;
use derivative::Derivative;
use futures::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::{io::AsyncWriteExt, process::Command, task::JoinHandle};
use ts_rs::TS;
use workspace_utils::{command_ext::GroupSpawnNoWindowExt, msg_store::MsgStore};

use crate::{
    approvals::ExecutorApprovalService,
    command::{CmdOverrides, CommandBuilder, apply_overrides},
    env::ExecutionEnv,
    executor_discovery::ExecutorDiscoveredOptions,
    executors::{
        AppendPrompt, AvailabilityInfo, BaseCodingAgent, ExecutorError, SpawnedChild,
        StandardCodingAgentExecutor,
    },
    logs::{
        ActionType, CommandExitStatus, CommandRunResult, NormalizedEntry, NormalizedEntryError,
        NormalizedEntryType, ToolStatus,
        utils::{ConversationPatch, EntryIndexProvider, shell_command_parsing::CommandCategory},
    },
    model_selector::{ModelInfo, ModelSelectorConfig, PermissionPolicy},
    profile::ExecutorConfig,
};

const KIRO_AUTH_REQUIRED_PATTERNS: &[&str] =
    &["Not logged in", "kiro-cli login", "Authentication required"];

#[derive(Derivative, Clone, Serialize, Deserialize, TS, JsonSchema)]
#[derivative(Debug, PartialEq)]
pub struct Kiro {
    #[serde(default)]
    pub append_prompt: AppendPrompt,

    #[serde(default, skip_serializing_if = "Option::is_none", alias = "model_id")]
    #[schemars(
        description = "Model override for Kiro chat, for example: auto, claude-opus-4.6, claude-sonnet-4.6, claude-opus-4.5, claude-sonnet-4.5, claude-sonnet-4, claude-haiku-4.5"
    )]
    pub model: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(
        description = "If true, passes --trust-all-tools so Kiro auto-approves tool usage; if false, Kiro prompts for tool approval"
    )]
    pub trust_all_tools: Option<bool>,

    #[serde(flatten)]
    pub cmd: CmdOverrides,

    #[serde(skip)]
    #[ts(skip)]
    #[derivative(Debug = "ignore", PartialEq = "ignore")]
    pub approvals: Option<Arc<dyn ExecutorApprovalService>>,
}

impl Kiro {
    fn build_command_builder(&self) -> Result<CommandBuilder, crate::command::CommandBuildError> {
        let mut builder = CommandBuilder::new("kiro-cli chat").extend_params(["--no-interactive"]);

        if let Some(model) = &self.model {
            builder = builder.extend_params(["--model", model.as_str()]);
        }

        if self.trust_all_tools.unwrap_or(true) {
            builder = builder.extend_params(["--trust-all-tools"]);
        }

        apply_overrides(builder, &self.cmd)
    }
}

#[async_trait]
impl StandardCodingAgentExecutor for Kiro {
    fn apply_overrides(&mut self, executor_config: &ExecutorConfig) {
        if let Some(model_id) = &executor_config.model_id {
            self.model = Some(model_id.clone());
        }
        if let Some(permission_policy) = executor_config.permission_policy.clone() {
            self.trust_all_tools = Some(matches!(permission_policy, PermissionPolicy::Auto));
        }
    }

    fn use_approvals(&mut self, approvals: Arc<dyn ExecutorApprovalService>) {
        self.approvals = Some(approvals);
    }

    async fn spawn(
        &self,
        current_dir: &Path,
        prompt: &str,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        let command_parts = self.build_command_builder()?.build_initial()?;
        let (executable_path, args) = command_parts.into_resolved().await?;
        let combined_prompt = self.append_prompt.combine_prompt(prompt);

        tracing::info!(
            "Kiro initial: Starting NEW session in {}, prompt length: {} chars",
            current_dir.display(),
            combined_prompt.len()
        );

        let mut command = Command::new(executable_path);
        command
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(current_dir)
            .args(&args);

        env.clone()
            .with_profile(&self.cmd)
            .apply_to_command(&mut command);

        let mut child = command.group_spawn_no_window()?;

        if let Some(mut stdin) = child.inner().stdin.take() {
            stdin.write_all(combined_prompt.as_bytes()).await?;
            stdin.shutdown().await?;
        }

        Ok(child.into())
    }

    async fn spawn_follow_up(
        &self,
        current_dir: &Path,
        prompt: &str,
        session_id: &str,
        _reset_to_message_id: Option<&str>,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        let command_parts = self
            .build_command_builder()?
            .build_follow_up(&["--resume".to_string()])?;
        let (executable_path, args) = command_parts.into_resolved().await?;

        let combined_prompt = self.append_prompt.combine_prompt(prompt);

        tracing::info!(
            "Kiro follow-up: RESUMING latest session in {}, requested session_id={}, prompt length: {} chars",
            current_dir.display(),
            session_id,
            combined_prompt.len()
        );

        let mut command = Command::new(executable_path);
        command
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(current_dir)
            .args(&args);

        env.clone()
            .with_profile(&self.cmd)
            .apply_to_command(&mut command);

        let mut child = command.group_spawn_no_window()?;

        if let Some(mut stdin) = child.inner().stdin.take() {
            stdin.write_all(combined_prompt.as_bytes()).await?;
            stdin.shutdown().await?;
        }

        Ok(child.into())
    }

    fn normalize_logs(
        &self,
        msg_store: Arc<MsgStore>,
        _worktree_path: &Path,
    ) -> Vec<JoinHandle<()>> {
        let entry_index_provider = EntryIndexProvider::start_from(&msg_store);

        let msg_store_stdout = msg_store.clone();
        let entry_index_provider_stdout = entry_index_provider.clone();
        let handle_stdout = tokio::spawn(async move {
            let mut stdout = msg_store_stdout.stdout_lines_stream();
            let mut assistant_buffer = String::new();
            let mut assistant_entry_id: Option<usize> = None;
            let mut tool_entries: HashMap<String, usize> = HashMap::new();

            while let Some(Ok(line)) = stdout.next().await {
                let content = strip_ansi_escapes::strip_str(&line);
                let trimmed = content.trim();
                if trimmed.is_empty() {
                    continue;
                }

                if let Some((tool_name, tool_content)) = detect_kiro_tool_start(trimmed) {
                    let entry = NormalizedEntry {
                        timestamp: None,
                        entry_type: NormalizedEntryType::ToolUse {
                            tool_name: tool_name.clone(),
                            action_type: kiro_tool_action(&tool_name, tool_content),
                            status: ToolStatus::Created,
                        },
                        content: tool_content.to_string(),
                        metadata: None,
                    };
                    let id = entry_index_provider_stdout.next();
                    tool_entries.insert(tool_name, id);
                    msg_store_stdout.push_patch(ConversationPatch::add_normalized_entry(id, entry));
                    continue;
                }

                if let Some((tool_name, exit_code)) = detect_kiro_tool_completion(trimmed)
                    && let Some(id) = tool_entries.remove(&tool_name)
                {
                    let entry = NormalizedEntry {
                        timestamp: None,
                        entry_type: NormalizedEntryType::ToolUse {
                            tool_name: tool_name.clone(),
                            action_type: kiro_tool_action_with_exit_code(
                                &tool_name, trimmed, exit_code,
                            ),
                            status: if exit_code == Some(0) {
                                ToolStatus::Success
                            } else {
                                ToolStatus::Failed
                            },
                        },
                        content: trimmed.to_string(),
                        metadata: None,
                    };
                    msg_store_stdout.push_patch(ConversationPatch::replace(id, entry));
                    continue;
                }

                assistant_buffer.push_str(trimmed);
                assistant_buffer.push('\n');
                let entry = NormalizedEntry {
                    timestamp: None,
                    entry_type: NormalizedEntryType::AssistantMessage,
                    content: assistant_buffer.trim_end().to_string(),
                    metadata: None,
                };
                if let Some(id) = assistant_entry_id {
                    msg_store_stdout.push_patch(ConversationPatch::replace(id, entry));
                } else {
                    let id = entry_index_provider_stdout.next();
                    assistant_entry_id = Some(id);
                    msg_store_stdout.push_patch(ConversationPatch::add_normalized_entry(id, entry));
                }
            }
        });

        let msg_store_stderr = msg_store.clone();
        let entry_index_provider_stderr = entry_index_provider;
        let handle_stderr = tokio::spawn(async move {
            let mut stderr = msg_store_stderr.stderr_chunked_stream();
            while let Some(Ok(chunk)) = stderr.next().await {
                let content = strip_ansi_escapes::strip_str(&chunk);
                let trimmed = content.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let entry = NormalizedEntry {
                    timestamp: None,
                    entry_type: if KIRO_AUTH_REQUIRED_PATTERNS
                        .iter()
                        .any(|pattern| trimmed.contains(pattern))
                    {
                        NormalizedEntryType::ErrorMessage {
                            error_type: NormalizedEntryError::SetupRequired,
                        }
                    } else {
                        NormalizedEntryType::ErrorMessage {
                            error_type: NormalizedEntryError::Other,
                        }
                    },
                    content: trimmed.to_string(),
                    metadata: None,
                };
                let id = entry_index_provider_stderr.next();
                msg_store_stderr.push_patch(ConversationPatch::add_normalized_entry(id, entry));
            }
        });

        vec![handle_stdout, handle_stderr]
    }

    fn default_mcp_config_path(&self) -> Option<std::path::PathBuf> {
        dirs::home_dir().map(|home| home.join(".kiro").join("settings").join("mcp.json"))
    }

    fn get_availability_info(&self) -> AvailabilityInfo {
        if which::which("kiro-cli").is_ok() {
            AvailabilityInfo::InstallationFound
        } else {
            AvailabilityInfo::NotFound
        }
    }

    async fn discover_options(
        &self,
        _workdir: Option<&std::path::Path>,
        _repo_path: Option<&std::path::Path>,
    ) -> Result<futures::stream::BoxStream<'static, json_patch::Patch>, ExecutorError> {
        use crate::logs::utils::patch;

        let options = ExecutorDiscoveredOptions {
            model_selector: ModelSelectorConfig {
                models: [
                    ("auto", "Auto"),
                    ("claude-opus-4.6", "Claude Opus 4.6"),
                    ("claude-sonnet-4.6", "Claude Sonnet 4.6"),
                    ("claude-opus-4.5", "Claude Opus 4.5"),
                    ("claude-sonnet-4.5", "Claude Sonnet 4.5"),
                    ("claude-sonnet-4", "Claude Sonnet 4"),
                    ("claude-haiku-4.5", "Claude Haiku 4.5"),
                ]
                .into_iter()
                .map(|(id, name)| ModelInfo {
                    id: id.to_string(),
                    name: name.to_string(),
                    provider_id: None,
                    reasoning_options: vec![],
                })
                .collect(),
                default_model: Some("auto".to_string()),
                permissions: vec![PermissionPolicy::Auto, PermissionPolicy::Supervised],
                ..Default::default()
            },
            ..Default::default()
        };
        Ok(Box::pin(futures::stream::once(async move {
            patch::executor_discovered_options(options)
        })))
    }

    fn get_preset_options(&self) -> ExecutorConfig {
        ExecutorConfig {
            executor: BaseCodingAgent::Kiro,
            variant: None,
            model_id: self.model.clone(),
            agent_id: None,
            reasoning_id: None,
            permission_policy: Some(if self.trust_all_tools.unwrap_or(true) {
                crate::model_selector::PermissionPolicy::Auto
            } else {
                crate::model_selector::PermissionPolicy::Supervised
            }),
        }
    }
}

fn detect_kiro_tool_start(line: &str) -> Option<(String, &str)> {
    let tool_name = line
        .split("(using tool:")
        .nth(1)?
        .trim_end_matches(')')
        .trim()
        .to_string();
    Some((tool_name, line))
}

fn detect_kiro_tool_completion(line: &str) -> Option<(String, Option<i32>)> {
    if line.starts_with('-') && line.contains("Completed in") {
        return Some(("shell".to_string(), Some(0)));
    }
    if line.starts_with("[ok]") {
        return Some(("shell".to_string(), Some(0)));
    }
    if line.starts_with("[error]") {
        return Some(("shell".to_string(), Some(1)));
    }
    None
}

fn kiro_tool_action(tool_name: &str, content: &str) -> ActionType {
    kiro_tool_action_with_exit_code(tool_name, content, Some(0))
}

fn kiro_tool_action_with_exit_code(
    tool_name: &str,
    content: &str,
    exit_code: Option<i32>,
) -> ActionType {
    match tool_name {
        "read" => ActionType::FileRead {
            path: content
                .split(':')
                .nth(1)
                .map(str::trim)
                .unwrap_or(content)
                .to_string(),
        },
        "shell" | "run_shell_command" => ActionType::CommandRun {
            command: content.to_string(),
            result: Some(CommandRunResult {
                exit_status: exit_code.map(|code| CommandExitStatus::ExitCode { code }),
                output: None,
            }),
            category: CommandCategory::from_command(content),
        },
        "web_fetch" => ActionType::WebFetch {
            url: content.to_string(),
        },
        "grep" | "glob" => ActionType::Search {
            query: content.to_string(),
        },
        _ => ActionType::Tool {
            tool_name: tool_name.to_string(),
            arguments: None,
            result: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_model_and_permission_overrides() {
        let mut executor = Kiro {
            append_prompt: AppendPrompt::default(),
            model: None,
            trust_all_tools: None,
            cmd: CmdOverrides::default(),
            approvals: None,
        };

        executor.apply_overrides(&ExecutorConfig {
            executor: BaseCodingAgent::Kiro,
            variant: None,
            model_id: Some("claude-sonnet-4.5".to_string()),
            agent_id: None,
            reasoning_id: None,
            permission_policy: Some(PermissionPolicy::Supervised),
        });

        assert_eq!(executor.model.as_deref(), Some("claude-sonnet-4.5"));
        assert_eq!(executor.trust_all_tools, Some(false));
    }

    #[test]
    fn exposes_supervised_preset_when_trust_is_disabled() {
        let executor = Kiro {
            append_prompt: AppendPrompt::default(),
            model: Some("auto".to_string()),
            trust_all_tools: Some(false),
            cmd: CmdOverrides::default(),
            approvals: None,
        };

        let preset = executor.get_preset_options();
        assert_eq!(preset.model_id.as_deref(), Some("auto"));
        assert_eq!(preset.permission_policy, Some(PermissionPolicy::Supervised));
    }

    #[test]
    fn detects_tool_start_from_plain_text_logs() {
        let parsed = detect_kiro_tool_start("Reading file: Cargo.toml (using tool: read)");
        assert_eq!(
            parsed,
            Some((
                "read".to_string(),
                "Reading file: Cargo.toml (using tool: read)",
            ))
        );
    }
}
