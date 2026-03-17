use db::models::{
    execution_process::{ExecutionProcess, ExecutionProcessRunReason},
    session::{CreateSession, Session},
    workspace::{Workspace, WorkspaceError},
};
use deployment::Deployment;
use executors::actions::ExecutorAction;
#[cfg(unix)]
use executors::{
    actions::{
        ExecutorActionType,
        script::{ScriptContext, ScriptRequest, ScriptRequestLanguage},
    },
    executors::kiro::Kiro,
};
use services::services::container::ContainerService;
use uuid::Uuid;

use crate::error::ApiError;

pub async fn run_kiro_setup(
    deployment: &crate::DeploymentImpl,
    workspace: &Workspace,
    _kiro: &Kiro,
) -> Result<ExecutionProcess, ApiError> {
    let latest_process = ExecutionProcess::find_latest_by_workspace_and_run_reason(
        &deployment.db().pool,
        workspace.id,
        &ExecutionProcessRunReason::CodingAgent,
    )
    .await?;

    let executor_action = if let Some(latest_process) = latest_process {
        let latest_action = latest_process
            .executor_action()
            .map_err(|e| ApiError::Workspace(WorkspaceError::ValidationError(e.to_string())))?;
        get_setup_helper_action()
            .await?
            .append_action(latest_action.to_owned())
    } else {
        get_setup_helper_action().await?
    };

    deployment
        .container()
        .ensure_container_exists(workspace)
        .await?;

    let session =
        match Session::find_latest_by_workspace_id(&deployment.db().pool, workspace.id).await? {
            Some(s) => s,
            None => {
                Session::create(
                    &deployment.db().pool,
                    &CreateSession {
                        executor: Some("kiro".to_string()),
                    },
                    Uuid::new_v4(),
                    workspace.id,
                )
                .await?
            }
        };

    let execution_process = deployment
        .container()
        .start_execution(
            workspace,
            &session,
            &executor_action,
            &ExecutionProcessRunReason::SetupScript,
        )
        .await?;
    Ok(execution_process)
}

#[cfg(unix)]
async fn get_setup_helper_action() -> Result<ExecutorAction, ApiError> {
    use utils::shell::UnixShell;

    let mut install_script = r#"#!/bin/bash
set -e

echo "Installing Kiro CLI..."

if ! command -v kiro-cli &> /dev/null; then
    curl -fsSL https://cli.kiro.dev/install | bash
    echo "Kiro CLI installed successfully"
else
    echo "Kiro CLI already installed"
fi
"#
    .to_string();

    if let Some(config_file) = UnixShell::current_shell().config_file() {
        let config_file_string = config_file.to_string_lossy().to_string();
        let quoted = shlex::try_quote(&config_file_string)
            .map_err(|e| ApiError::Workspace(WorkspaceError::ValidationError(e.to_string())))?;
        install_script.push_str(&format!(
            r#"
echo "Ensuring Kiro CLI is on PATH..."
echo 'export PATH="$HOME/.local/bin:$PATH"' >> {quoted}
"#
        ));
    }

    let install_request = ScriptRequest {
        script: install_script,
        language: ScriptRequestLanguage::Bash,
        context: ScriptContext::ToolInstallScript,
        working_dir: None,
    };

    let login_request = ScriptRequest {
        script: r#"#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
kiro-cli login
kiro-cli whoami --format json || true
"#
        .to_string(),
        language: ScriptRequestLanguage::Bash,
        context: ScriptContext::ToolInstallScript,
        working_dir: None,
    };

    Ok(ExecutorAction::new(
        ExecutorActionType::ScriptRequest(install_request),
        Some(Box::new(ExecutorAction::new(
            ExecutorActionType::ScriptRequest(login_request),
            None,
        ))),
    ))
}

#[cfg(not(unix))]
async fn get_setup_helper_action() -> Result<ExecutorAction, ApiError> {
    Err(ApiError::Executor(
        executors::executors::ExecutorError::SetupHelperNotSupported,
    ))
}
