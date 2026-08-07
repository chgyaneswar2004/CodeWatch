"""
demo gitlab api server
"""

import asyncio
import logging
import time
import traceback
from typing import Callable

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from gitlab import Gitlab
from gitlab.v4.objects import ProjectMergeRequest
from langchain_community.callbacks.manager import get_openai_callback
from pydantic import BaseModel

from codedog.actors.reporters.pull_request import PullRequestReporter
from codedog.chains.code_review.base import CodeReviewChain
from codedog.chains.pr_summary.base import PRSummaryChain
from codedog.retrievers.gitlab_retriever import GitlabRetriever
from codedog.utils.langchain_utils import load_model_by_name
from codedog.utils.email_utils import send_report_email
from codedog.version import VERSION
from codedog.config.settings import settings

# config
host = "127.0.0.1"
port = 32167
worker_num = 1
gitlab_token = settings.gitlab_token or "your gitlab token here"
gitlab_base_url = settings.gitlab_url or "your gitlab base url here"
gitlab_webhook_token = settings.gitlab_webhook_token

# fastapi
app = FastAPI()


class GitlabEvent(BaseModel):
    object_kind: str
    project: dict
    object_attributes: dict


@app.post("/gitlab_event", response_class=PlainTextResponse)
async def gitlab_event(event: GitlabEvent, x_gitlab_token: str = Header(None)) -> str:
    """Gitlab webhook."""
    if gitlab_webhook_token:
        if not x_gitlab_token or x_gitlab_token != gitlab_webhook_token:
            raise HTTPException(status_code=401, detail="Invalid or missing X-Gitlab-Token")
    else:
        logging.warning("GitLab webhook token verification is disabled (GITLAB_WEBHOOK_TOKEN is not set)")

    t = time.time()
    status = "success"

    try:
        message = await handle_gitlab_event(event)
    except Exception:
        logging.warning(
            "Fail to handle gitlab event: %s",
            traceback.format_exc().replace("\n", "\\n"),
        )
        message = "fail to handle event"
        status = "failed"

    logging.info(
        "Handle gitlab event kind: %s, duration: %f, status: %s",
        event.object_kind,
        time.time() - t,
        status,
    )
    return message


async def handle_gitlab_event(event: GitlabEvent) -> str:
    """Trigger merge request review based on gitlab event."""
    if not _validate_event(event):
        raise ValueError("Invalid Event.")

    project_id = event.project["id"]
    merge_request_iid = event.object_attributes["iid"]

    logging.info("Start review merge request %s %d", project_id, merge_request_iid)

    retriever = GitlabRetriever(
        client=Gitlab(gitlab_base_url, private_token=gitlab_token),
        project_name_or_id=project_id,
        merge_request_iid=merge_request_iid,
    )
    callback = _comment_callback(retriever._git_merge_request)

    asyncio.create_task(handle_event(retriever, callback=callback))
    return "Review Request Submitted."


def _validate_event(event: GitlabEvent) -> bool:
    if event.object_kind != "merge_request":
        return False
    # trigger review when PR is opened or new commit is pushed (update action)
    action = event.object_attributes.get("action")
    if action not in ["open", "update"]:
        return False
    return True


def _comment_callback(merge_request: ProjectMergeRequest) -> Callable[[str], None]:
    """Build callback function for merge request comment."""

    def callback(report: str):
        merge_request.notes.create(
            {
                "body": report,
                "project_id": merge_request.project_id,
                "merge_request_iid": merge_request.iid,
            }
        )

    return callback


async def handle_event(retriever: GitlabRetriever, callback: Callable):
    t = time.time()
    summary_chain = PRSummaryChain.from_llm(
        code_summary_llm=load_model_by_name(settings.code_summary_model),
        pr_summary_llm=load_model_by_name(settings.pr_summary_model)
    )
    review_chain = CodeReviewChain.from_llm(llm=load_model_by_name(settings.code_review_model))

    with get_openai_callback() as cb:
        summary_result = await summary_chain.ainvoke({"pull_request": retriever.pull_request})
        review_result = await review_chain.ainvoke({"pull_request": retriever.pull_request})
        reporter = PullRequestReporter(
            pr_summary=summary_result["pr_summary"],
            code_summaries=summary_result["code_summaries"],
            pull_request=retriever.pull_request,
            code_reviews=review_result["code_reviews"],
            telemetry={
                "start_time": t,
                "time_usage": time.time() - t,
                "cost": cb.total_cost,
                "tokens": cb.total_tokens,
            },
        )
        report = reporter.report()
        await asyncio.to_thread(callback, report)

        # Send email report if configured
        if settings.email_enabled and settings.notification_emails:
            email_addresses = [email.strip() for email in settings.notification_emails.split(",") if email.strip()]
            if email_addresses:
                logging.info(f"Sending MR review report email to {', '.join(email_addresses)}")
                subject = f"[CodeDog Webhook] MR #{retriever.pull_request.iid} Review: {retriever.pull_request.title}"
                try:
                    await asyncio.to_thread(
                        send_report_email,
                        to_emails=email_addresses,
                        subject=subject,
                        markdown_content=report,
                    )
                    logging.info("MR review report email sent successfully.")
                except Exception as e:
                    logging.error(f"Failed to send MR review report email: {e}")


def start():
    uvicorn.run("examples.gitlab_server:app", host=host, port=port, workers=worker_num)
    logging.info(f"Codedog v{VERSION}: server start.")


if __name__ == "__main__":
    start()
