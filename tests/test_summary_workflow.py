"""
Tests for .github/workflows/summary.yml

Validates the structure, security properties, and configuration of the
"Summarize new issues" GitHub Actions workflow.
"""

import re
from pathlib import Path

import pytest
import yaml

WORKFLOW_PATH = Path(__file__).parent.parent / ".github" / "workflows" / "summary.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    """
    Load and parse the repository's summary GitHub Actions workflow YAML.
    
    Returns:
        dict: Parsed YAML content of the workflow file at WORKFLOW_PATH.
    """
    with open(WORKFLOW_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def summary_job(workflow: dict) -> dict:
    """
    Return the mapping for the `summary` job from a parsed GitHub Actions workflow.
    
    Parameters:
        workflow (dict): Parsed workflow structure (as loaded by `yaml.safe_load`) representing the top-level workflow document.
    
    Returns:
        dict: The dictionary value of `workflow["jobs"]["summary"]`.
    """
    return workflow["jobs"]["summary"]


@pytest.fixture(scope="module")
def steps(summary_job: dict) -> list:
    """
    Return the list of steps from a workflow job definition.
    
    Parameters:
    	summary_job (dict): Parsed job mapping from the workflow YAML (expected to contain a "steps" key).
    
    Returns:
    	steps (list): The list of step dictionaries found under the job's "steps" key.
    """
    return summary_job["steps"]


@pytest.fixture(scope="module")
def checkout_step(steps: list) -> dict:
    """
    Return the first step from a job's steps list, expected to be the repository checkout step.
    
    Parameters:
        steps (list): Sequence of step dictionaries from a workflow job's "steps" field.
    
    Returns:
        dict: The first step dictionary (checkout step).
    """
    return steps[0]


@pytest.fixture(scope="module")
def inference_step(steps: list) -> dict:
    """
    Retrieve the inference (AI) step from a workflow's steps list.
    
    Parameters:
        steps (list): List of step mappings as parsed from the workflow job's `steps` sequence.
    
    Returns:
        dict: The step dictionary at index 1 representing the AI inference step.
    """
    return steps[1]


@pytest.fixture(scope="module")
def comment_step(steps: list) -> dict:
    """
    Get the comment step (third step) from a job's steps list.
    
    Parameters:
        steps (list): List of step mappings as parsed from the workflow job's `steps`.
    
    Returns:
        dict: The third step dictionary, expected to be the comment step.
    """
    return steps[2]


# ---------------------------------------------------------------------------
# Workflow-level tests
# ---------------------------------------------------------------------------

class TestWorkflowMetadata:
    def test_workflow_file_exists(self):
        assert WORKFLOW_PATH.exists(), f"Workflow file not found at {WORKFLOW_PATH}"

    def test_workflow_name(self, workflow):
        assert workflow["name"] == "Summarize new issues"

    def test_workflow_is_valid_yaml(self):
        """Verify the file parses as valid YAML without error."""
        with open(WORKFLOW_PATH) as f:
            parsed = yaml.safe_load(f)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Trigger tests
# ---------------------------------------------------------------------------

class TestWorkflowTrigger:
    # NOTE: PyYAML parses the YAML key `on` as the Python boolean True,
    # because `on` is a legacy YAML 1.1 boolean alias.  We therefore access
    # the trigger block via workflow[True].

    def test_trigger_is_issues(self, workflow):
        trigger = workflow[True]
        assert "issues" in trigger, "Workflow must trigger on 'issues' events"

    def test_trigger_type_is_opened(self, workflow):
        """
        Verify the workflow's issues trigger includes the "opened" event.
        
        Parameters:
            workflow (dict): Parsed YAML object for the workflow file.
        """
        types = workflow[True]["issues"]["types"]
        assert "opened" in types, "Workflow must trigger on issues.opened"

    def test_trigger_only_opened_type(self, workflow):
        """Workflow should only react to 'opened', not other issue events."""
        types = workflow[True]["issues"]["types"]
        assert types == ["opened"], (
            f"Workflow should trigger only on 'opened', got: {types}"
        )

    def test_no_extra_triggers(self, workflow):
        """Only the 'issues' trigger should be present."""
        trigger = workflow[True]
        assert list(trigger.keys()) == ["issues"], (
            f"Unexpected extra triggers: {list(trigger.keys())}"
        )


# ---------------------------------------------------------------------------
# Job configuration tests
# ---------------------------------------------------------------------------

class TestJobConfiguration:
    def test_summary_job_exists(self, workflow):
        assert "summary" in workflow["jobs"]

    def test_only_one_job(self, workflow):
        """
        Asserts the workflow defines exactly one job.
        
        If the assertion fails, the error message includes the names of the jobs found.
        """
        assert len(workflow["jobs"]) == 1, (
            f"Expected exactly 1 job, found: {list(workflow['jobs'].keys())}"
        )

    def test_runs_on_ubuntu_latest(self, summary_job):
        assert summary_job["runs-on"] == "ubuntu-latest"


# ---------------------------------------------------------------------------
# Permission tests
# ---------------------------------------------------------------------------

class TestPermissions:
    def test_issues_write_permission(self, summary_job):
        assert summary_job["permissions"]["issues"] == "write"

    def test_models_read_permission(self, summary_job):
        assert summary_job["permissions"]["models"] == "read"

    def test_contents_read_permission(self, summary_job):
        assert summary_job["permissions"]["contents"] == "read"

    def test_no_extra_permissions(self, summary_job):
        """Permissions should be least-privilege — no unexpected grants."""
        expected = {"issues", "models", "contents"}
        actual = set(summary_job["permissions"].keys())
        assert actual == expected, f"Unexpected permissions: {actual - expected}"

    def test_no_write_all_permissions(self, summary_job):
        """The job must not use 'write-all' or blanket permission grants."""
        perms = summary_job["permissions"]
        assert perms != "write-all"
        assert perms != "read-all"
        for value in perms.values():
            assert value != "write-all"


# ---------------------------------------------------------------------------
# Step structure tests
# ---------------------------------------------------------------------------

class TestStepStructure:
    def test_exactly_three_steps(self, steps):
        assert len(steps) == 3, f"Expected 3 steps, got {len(steps)}"

    def test_checkout_is_first_step(self, checkout_step):
        assert checkout_step.get("uses", "").startswith("actions/checkout")

    def test_inference_is_second_step(self, inference_step):
        assert inference_step.get("uses", "").startswith("actions/ai-inference")

    def test_comment_is_third_step(self, comment_step):
        assert "gh issue comment" in comment_step.get("run", "")


# ---------------------------------------------------------------------------
# Checkout step tests
# ---------------------------------------------------------------------------

class TestCheckoutStep:
    def test_uses_checkout_v4(self, checkout_step):
        assert checkout_step["uses"] == "actions/checkout@v4"

    def test_checkout_step_name(self, checkout_step):
        assert checkout_step["name"] == "Checkout repository"


# ---------------------------------------------------------------------------
# AI inference step tests
# ---------------------------------------------------------------------------

class TestInferenceStep:
    def test_uses_ai_inference_v1(self, inference_step):
        assert inference_step["uses"] == "actions/ai-inference@v1"

    def test_has_id_inference(self, inference_step):
        """
        Verify the inference step is assigned the id "inference".
        
        Asserts that the `inference_step` fixture has an `"id"` key with the value `"inference"`.
        """
        assert inference_step.get("id") == "inference"

    def test_has_prompt(self, inference_step):
        assert "prompt" in inference_step.get("with", {}), "Inference step must have a 'prompt' input"

    def test_prompt_contains_issue_title(self, inference_step):
        prompt = inference_step["with"]["prompt"]
        assert "${{ github.event.issue.title }}" in prompt

    def test_prompt_contains_issue_body(self, inference_step):
        prompt = inference_step["with"]["prompt"]
        assert "${{ github.event.issue.body }}" in prompt

    def test_prompt_warns_about_untrusted_content(self, inference_step):
        """Prompt must instruct the model that issue content is untrusted."""
        prompt = inference_step["with"]["prompt"].lower()
        assert "untrusted" in prompt, (
            "Prompt must warn the model that issue title/body are untrusted"
        )

    def test_prompt_instructs_not_to_follow_instructions(self, inference_step):
        """
        Asserts the inference prompt instructs the model not to follow user-supplied instructions.
        
        Checks that the prompt (case-insensitive) contains either "do not follow instructions" or "not follow instructions".
        """
        prompt = inference_step["with"]["prompt"].lower()
        assert "do not follow instructions" in prompt or "not follow instructions" in prompt

    def test_prompt_asks_for_summary(self, inference_step):
        prompt = inference_step["with"]["prompt"].lower()
        assert "summariz" in prompt  # covers "summarize" / "summarizing"

    def test_inference_step_name(self, inference_step):
        assert inference_step["name"] == "Run AI inference"


# ---------------------------------------------------------------------------
# Comment step tests
# ---------------------------------------------------------------------------

class TestCommentStep:
    def test_comment_step_name(self, comment_step):
        assert comment_step["name"] == "Comment with AI summary"

    def test_uses_gh_cli(self, comment_step):
        run_script = comment_step.get("run", "")
        assert "gh issue comment" in run_script

    def test_run_references_issue_number_via_env(self, comment_step):
        """ISSUE_NUMBER should be passed via env var, not inline expression."""
        run_script = comment_step.get("run", "")
        assert "$ISSUE_NUMBER" in run_script, (
            "gh command must reference ISSUE_NUMBER via env var for safety"
        )

    def test_run_references_response_via_env(self, comment_step):
        """
        Assert the comment step's run script references the inference response via the $RESPONSE environment variable.
        
        This test ensures the GitHub CLI command uses the environment variable (``$RESPONSE``) rather than inlining the inference output directly into the script.
        """
        run_script = comment_step.get("run", "")
        assert "$RESPONSE" in run_script, (
            "gh command must reference RESPONSE via env var for safety"
        )

    def test_run_does_not_inline_github_expressions(self, comment_step):
        """
        The run script must NOT contain raw ${{ }} expressions.
        Injecting expressions directly into shell scripts is a security risk
        (script injection); they should always be passed through env vars.
        """
        run_script = comment_step.get("run", "")
        assert "${{" not in run_script, (
            "run script must not contain raw ${{ }} expressions — use env vars instead"
        )

    def test_gh_token_uses_github_token_secret(self, comment_step):
        env = comment_step.get("env", {})
        assert env.get("GH_TOKEN") == "${{ secrets.GITHUB_TOKEN }}"

    def test_issue_number_env_references_event(self, comment_step):
        env = comment_step.get("env", {})
        assert env.get("ISSUE_NUMBER") == "${{ github.event.issue.number }}"

    def test_response_env_references_inference_output(self, comment_step):
        env = comment_step.get("env", {})
        assert env.get("RESPONSE") == "${{ steps.inference.outputs.response }}"

    def test_env_has_exactly_three_vars(self, comment_step):
        env = comment_step.get("env", {})
        assert set(env.keys()) == {"GH_TOKEN", "ISSUE_NUMBER", "RESPONSE"}, (
            f"Unexpected env vars in comment step: {set(env.keys())}"
        )


# ---------------------------------------------------------------------------
# Security / regression tests
# ---------------------------------------------------------------------------

class TestSecurityProperties:
    def test_no_raw_expressions_in_any_run_script(self, steps):
        """
        Ensure no step's shell `run` script contains literal GitHub expression delimiters.
        
        Asserts that the substring "${{" does not appear in any step's `run` value to prevent inlining `${{ ... }}` expressions into shell scripts.
        """
        for step in steps:
            run_script = step.get("run", "")
            assert "${{" not in run_script, (
                f"Step '{step.get('name', '?')}' contains raw ${{{{ }}}} in run script"
            )

    def test_no_secrets_hardcoded_in_workflow(self):
        """
        Ensure the workflow file contains no hardcoded GitHub personal access tokens.
        
        Searches the raw workflow text for token-like strings matching the pattern `ghp_[A-Za-z0-9]+` that are not part of a GitHub Actions expression (i.e., not preceded by `${{`) and fails the test if any such occurrences are found.
        """
        raw_content = WORKFLOW_PATH.read_text()
        # Look for patterns that suggest hardcoded tokens (not the expression syntax)
        suspicious = re.findall(r'(?<!\$\{\{)\bghp_[A-Za-z0-9]+\b', raw_content)
        assert not suspicious, f"Possible hardcoded GitHub token found: {suspicious}"

    def test_gh_token_not_exposed_in_prompt(self, inference_step):
        """The AI prompt must not reference secrets or tokens."""
        prompt = inference_step["with"]["prompt"]
        assert "secrets." not in prompt
        assert "GITHUB_TOKEN" not in prompt
        assert "GH_TOKEN" not in prompt

    def test_prompt_injection_defense_present(self, inference_step):
        """
        The prompt must include a defense against prompt injection attacks
        embedded in issue titles or bodies.
        """
        prompt = inference_step["with"]["prompt"]
        defense_keywords = ["untrusted", "malicious", "do not follow"]
        assert any(kw in prompt.lower() for kw in defense_keywords), (
            "Prompt must contain injection defense language"
        )

    def test_models_permission_is_read_not_write(self, summary_job):
        """Models permission should be read-only."""
        assert summary_job["permissions"]["models"] == "read"

    def test_issues_permission_is_write_not_admin(self, summary_job):
        """
        Assert that the job's 'issues' permission is set to "write".
        """
        assert summary_job["permissions"]["issues"] == "write"