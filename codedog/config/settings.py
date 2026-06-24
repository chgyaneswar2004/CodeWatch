import os
import json
from typing import Set, Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environmental variables from .env file if present
load_dotenv()

class Settings(BaseModel):
    # Git configurations
    github_token: str = Field(default_factory=lambda: os.environ.get("GITHUB_TOKEN", ""))
    github_webhook_secret: str = Field(default_factory=lambda: os.environ.get("GITHUB_WEBHOOK_SECRET", ""))
    gitlab_token: str = Field(default_factory=lambda: os.environ.get("GITLAB_TOKEN", ""))
    gitlab_url: str = Field(default_factory=lambda: os.environ.get("GITLAB_URL") or os.environ.get("GITLAB_BASE_URL", "https://gitlab.com"))
    gitlab_webhook_token: str = Field(default_factory=lambda: os.environ.get("GITLAB_WEBHOOK_TOKEN", ""))

    # Model selections
    code_summary_model: str = Field(default_factory=lambda: os.environ.get("CODE_SUMMARY_MODEL", "gpt-3.5"))
    pr_summary_model: str = Field(default_factory=lambda: os.environ.get("PR_SUMMARY_MODEL", "gpt-4"))
    code_review_model: str = Field(default_factory=lambda: os.environ.get("CODE_REVIEW_MODEL", "gpt-3.5"))

    # LLM Models
    gpt35_model: str = Field(default_factory=lambda: os.environ.get("GPT35_MODEL", "gpt-3.5-turbo"))
    gpt4_model: str = Field(default_factory=lambda: os.environ.get("GPT4_MODEL", "gpt-4"))
    gpt4o_model: str = Field(default_factory=lambda: os.environ.get("GPT4O_MODEL", "gpt-4o"))

    # Azure OpenAI
    azure_openai: bool = Field(default_factory=lambda: os.environ.get("AZURE_OPENAI", "").lower() in ("true", "1", "yes"))
    azure_openai_deployment_id: str = Field(default_factory=lambda: os.environ.get("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-35-turbo"))
    azure_openai_gpt4_deployment_id: Optional[str] = Field(default_factory=lambda: os.environ.get("AZURE_OPENAI_GPT4_DEPLOYMENT_ID"))
    azure_openai_gpt4o_deployment_id: Optional[str] = Field(default_factory=lambda: os.environ.get("AZURE_OPENAI_GPT4O_DEPLOYMENT_ID"))
    azure_openai_api_key: str = Field(default_factory=lambda: os.environ.get("AZURE_OPENAI_API_KEY", ""))
    azure_openai_api_base: str = Field(default_factory=lambda: os.environ.get("AZURE_OPENAI_API_BASE", ""))

    # OpenAI
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    openai_api_base: Optional[str] = Field(default_factory=lambda: os.environ.get("OPENAI_API_BASE"))

    # DeepSeek
    deepseek_api_key: Optional[str] = Field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY"))
    deepseek_model: Optional[str] = Field(default_factory=lambda: os.environ.get("DEEPSEEK_MODEL"))
    deepseek_api_base: Optional[str] = Field(default_factory=lambda: os.environ.get("DEEPSEEK_API_BASE"))
    deepseek_temperature: float = Field(default_factory=lambda: float(os.environ.get("DEEPSEEK_TEMPERATURE", "0")))
    deepseek_max_tokens: int = Field(default_factory=lambda: int(os.environ.get("DEEPSEEK_MAX_TOKENS", "4096")))
    deepseek_top_p: float = Field(default_factory=lambda: float(os.environ.get("DEEPSEEK_TOP_P", "0.95")))
    deepseek_timeout: int = Field(default_factory=lambda: int(os.environ.get("DEEPSEEK_TIMEOUT", "600")))
    deepseek_max_retries: int = Field(default_factory=lambda: int(os.environ.get("DEEPSEEK_MAX_RETRIES", "3")))
    deepseek_retry_delay: int = Field(default_factory=lambda: int(os.environ.get("DEEPSEEK_RETRY_DELAY", "5")))

    # DeepSeek R1
    deepseek_r1_model: Optional[str] = Field(default_factory=lambda: os.environ.get("DEEPSEEK_R1_MODEL"))
    deepseek_r1_api_base: Optional[str] = Field(default_factory=lambda: os.environ.get("DEEPSEEK_R1_API_BASE"))

    # SMTP
    smtp_server: Optional[str] = Field(default_factory=lambda: os.environ.get("SMTP_SERVER"))
    smtp_port: int = Field(default_factory=lambda: int(os.environ.get("SMTP_PORT", 587)))
    smtp_username: Optional[str] = Field(default_factory=lambda: os.environ.get("SMTP_USERNAME"))
    smtp_password: Optional[str] = Field(default_factory=lambda: os.environ.get("CODEDOG_SMTP_PASSWORD") or os.environ.get("SMTP_PASSWORD"))
    email_enabled: bool = Field(default_factory=lambda: os.environ.get("EMAIL_ENABLED", "").lower() in ("true", "1", "yes"))
    notification_emails: str = Field(default_factory=lambda: os.environ.get("NOTIFICATION_EMAILS", ""))

    # Code Suffixes
    codedog_support_code_file_suffix: str = Field(default_factory=lambda: os.environ.get("CODEDOG_SUPPORT_CODE_FILE_SUFFIX", "py,java,go,js,ts,php,c,cpp,h,cs,rs"))
    codedog_suffix_language_mapping: str = Field(default_factory=lambda: os.environ.get("CODEDOG_SUFFIX_LANGUAGE_MAPPING", ""))

    # Dev Eval
    dev_eval_default_include: str = Field(default_factory=lambda: os.environ.get("DEV_EVAL_DEFAULT_INCLUDE", ""))
    dev_eval_default_exclude: str = Field(default_factory=lambda: os.environ.get("DEV_EVAL_DEFAULT_EXCLUDE", ""))

    @property
    def support_code_file_suffix_set(self) -> Set[str]:
        return set([ext.strip().lower() for ext in self.codedog_support_code_file_suffix.split(",") if ext.strip()])

    @property
    def suffix_language_mapping_dict(self) -> Dict[str, str]:
        # Default mapping
        mapping = {
            "py": "python",
            "java": "java",
            "go": "go",
            "js": "javascript",
            "ts": "typescript",
            "php": "php",
            "c": "c",
            "cpp": "cpp",
            "h": "c",
            "cs": "csharp",
            "rs": "rust",
        }
        if self.codedog_suffix_language_mapping:
            try:
                # Support JSON format or comma-separated "ext:lang" format
                if self.codedog_suffix_language_mapping.strip().startswith("{"):
                    mapping.update(json.loads(self.codedog_suffix_language_mapping))
                else:
                    for item in self.codedog_suffix_language_mapping.split(","):
                        if ":" in item:
                            k, v = item.split(":", 1)
                            mapping[k.strip().lower()] = v.strip().lower()
            except Exception:
                pass

        # Ensure default/unmapped suffixes are mapped to themselves
        for ext in self.support_code_file_suffix_set:
            if ext not in mapping:
                mapping[ext] = ext
        return mapping

# Singleton instance
settings = Settings()
