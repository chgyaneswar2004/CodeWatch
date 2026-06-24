"""
demo github api server
"""

import asyncio
import logging
import time
import os
import hmac
import hashlib

import uvicorn
from fastapi import FastAPI, Request, Header, HTTPException
from github import Github
from langchain_community.callbacks.manager import get_openai_callback
from pydantic import BaseModel

from codedog.actors.reporters.pull_request import PullRequestReporter
from codedog.chains.code_review.base import CodeReviewChain
from codedog.chains.pr_summary.base import PRSummaryChain
from codedog.retrievers.github_retriever import GithubRetriever
from codedog.utils.langchain_utils import load_model_by_name
from codedog.version import VERSION
from codedog.config.settings import settings

# config
host = "127.0.0.1"
port = 32167
worker_num = 1
github_token = settings.github_token or "your github token here"
github_webhook_secret = settings.github_webhook_secret


# fastapi
app = FastAPI()


class GithubEvent(BaseModel):
    action: str
    number: int
    pull_request: dict
    repository: dict


@app.post("/github")
async def github(request: Request, event: GithubEvent, x_hub_signature_256: str = Header(None)):
    """Github webhook.

    Args:
        request (Request): FastAPI request.
        event (GithubEvent): Github event.
        x_hub_signature_256 (str): GitHub webhook signature.
    Returns:
        Response: message.
    """
    if github_webhook_secret:
        if not x_hub_signature_256:
            raise HTTPException(status_code=401, detail="X-Hub-Signature-256 header is missing")
        body = await request.body()
        signature = "sha256=" + hmac.new(
            github_webhook_secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, x_hub_signature_256):
            raise HTTPException(status_code=401, detail="Invalid signature")
    else:
        logging.warning("GitHub webhook signature verification is disabled (GITHUB_WEBHOOK_SECRET is not set)")

    try:
        message = await handle_github_event(event)
    except Exception as e:
        logging.error(f"Error handling event: {str(e)}")
        return str(e)
    return message


async def handle_github_event(event: GithubEvent, **kwargs) -> str:
    _github_event_filter(event)

    repository_id: int = event.repository.get("id", 0)
    pull_request_number: int = event.number

    logging.info(
        f"Retrieve pull request from Github {repository_id} {pull_request_number}"
    )

    asyncio.create_task(handle_pull_request(repository_id, pull_request_number, **kwargs))

    return "Review Submitted."


async def handle_pull_request(
    repository_id: int,
    pull_request_number: int,
    local=False,
    language="en",
    **kwargs,
):
    t = time.time()
    client = Github(github_token)
    retriever = GithubRetriever(
        client=client,
        repository_name_or_id=repository_id,
        pull_request_number=pull_request_number,
    )
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
            language=language,
        )
        report = reporter.report()
        if local:
            print(report)
        else:
            await asyncio.to_thread(retriever._git_pull_request.create_issue_comment, report)


def _github_event_filter(event: GithubEvent):
    """filter github event.

    Args:
        event (GithubEvent): github event.

    Returns:
        bool: True if the event is filtered.
    """
    pull_request = event.pull_request

    if not pull_request:
        raise RuntimeError("Not a pull request event.")
    if event.action not in ("opened"):
        raise RuntimeError("Not a pull request open event.")
    if pull_request.get("state", "") != "open":
        raise RuntimeError("Pull request status is not open.")
    if pull_request.get("draft", False):
        raise RuntimeError("Pull request is a draft")


def start():
    uvicorn.run("examples.github_server:app", host=host, port=port, workers=worker_num)
    logging.info(f"Codedog v{VERSION}: server start.")


if __name__ == "__main__":
    start()

