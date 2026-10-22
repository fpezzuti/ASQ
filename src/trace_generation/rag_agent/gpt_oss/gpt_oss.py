import json
import os
import asyncio
from typing import Any

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
from openai import OpenAI

from .parse import ParsedModelOutput, parse_chat_completion


def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r") as f:
        return f.read()
    
QA_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "gpt_prompt_qa.txt")


def _format_docs(entries: list[dict]) -> str:
    lines = []
    for i, item in enumerate(entries, start=1):
        title = item.get("title")
        content = item.get("content", "")
        if title:
            lines.append(f"Doc {i}(Title: {title}) {content}")
        else:
            lines.append(f"Doc {i}: {content}")
    return "\n".join(lines) + ("\n" if lines else "")


class _TerrierOpenAIBackend:
    def __init__(self, model_id: str, max_new_tokens: int = 512, max_retries: int = 3, retry_delay_s: int = 60):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("TERRIER_API_KEY")
        if not api_key:
            raise ValueError("gpt_oss requires an API key. Export OPENAI_API_KEY or TERRIER_API_KEY before running.")
        self.client = OpenAI(base_url="http://api.terrier.org/v1", api_key=api_key)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.max_retries = max(1, int(max_retries))
        self.retry_delay_s = max(0, int(retry_delay_s))

    @staticmethod
    def _is_starting_up_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        print("Error from model generation:", msg)
        return "gpt-oss-20b is starting up" in msg or ("error code: 503" in msg and "starting up" in msg)

    def generate(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_new_tokens,
            tools=[SEARCH_TOOL],
            tool_choice="auto",
        )
        return response

    async def generate_async(self, messages: list[dict]) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await asyncio.to_thread(self.generate, messages)
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries or not self._is_starting_up_error(exc):
                    raise
                print(
                    f"Model {self.model_id} is still starting up; "
                    f"retrying in {self.retry_delay_s}s ({attempt}/{self.max_retries}). "
                    f"Error: {exc}"
                )
                await asyncio.sleep(self.retry_delay_s)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("generate_async failed without an exception.")


def _search(
    search_pipeline: pt.Transformer,
    query: str,
    qid: str,
) -> list[dict]:
    inp = pd.DataFrame([{"qid": qid, "query": query}])
    res = search_pipeline(inp)

    entries = []
    for row in res.itertuples():
        docid = str(row.docno)
        title = getattr(row, "title", None)
        content = getattr(row, "text", "")
        rank = getattr(row, "rank", None)
        score = getattr(row, "score", None)
        first_stage_rank = getattr(row, "first_stage_rank", rank)
        first_stage_score = getattr(row, "first_stage_score", score)
        entries.append(
            {
                "docid": docid,
                "title": title,
                "content": content,
                "rank": rank,
                "score": score,
                "first_stage_rank": first_stage_rank,
                "first_stage_score": first_stage_score,
            }
        )
    return entries


def _postprocess_search_results(res: pd.DataFrame, qid: str) -> list[dict]:
    entries = []
    for row in res.itertuples():
        docid = str(row.docno)
        title = getattr(row, "title", None)
        content = getattr(row, "text", "")
        rank = getattr(row, "rank", None)
        score = getattr(row, "score", None)
        first_stage_rank = getattr(row, "first_stage_rank", rank)
        first_stage_score = getattr(row, "first_stage_score", score)
        entries.append(
            {
                "docid": docid,
                "title": title,
                "content": content,
                "rank": rank,
                "score": score,
                "first_stage_rank": first_stage_rank,
                "first_stage_score": first_stage_score,
            }
        )
    return entries


class _RetrieverBatcher:
    def __init__(self, search_pipeline: pt.Transformer, max_batch_size: int = 1, max_wait_ms: int = 25):
        self.search_pipeline = search_pipeline
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_s = max(0, int(max_wait_ms)) / 1000.0
        self._pending: list[tuple[str, str, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

    async def submit(self, qid: str, query: str) -> list[dict]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        req = (qid, query, fut)
        async with self._lock:
            self._pending.append(req)
            if len(self._pending) >= self.max_batch_size:
                await self._flush_locked()
            elif self._flush_task is None:
                self._flush_task = asyncio.create_task(self._delayed_flush())
        return await fut

    async def _delayed_flush(self) -> None:
        await asyncio.sleep(self.max_wait_s)
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        if self._flush_task is not None:
            self._flush_task.cancel()
            self._flush_task = None
        if not self._pending:
            return
        pending = self._pending
        self._pending = []
        qids = [qid for qid, _, _ in pending]
        queries = [query for _, query, _ in pending]
        inp = pd.DataFrame({"qid": qids, "query": queries})
        try:
            batched_res = await asyncio.to_thread(self.search_pipeline, inp)
        except Exception as exc:
            for qid, query, fut in pending:
                if fut.done():
                    continue
                try:
                    single_inp = pd.DataFrame([{"qid": qid, "query": query}])
                    single_res = await asyncio.to_thread(self.search_pipeline, single_inp)
                    entries = _postprocess_search_results(single_res, qid=qid)
                    fut.set_result(entries)
                except Exception as single_exc:
                    fut.set_exception(single_exc)
            return

        grouped = {}
        if isinstance(batched_res, pd.DataFrame) and not batched_res.empty and "qid" in batched_res.columns:
            grouped = {str(k): v.copy() for k, v in batched_res.groupby("qid", sort=False)}

        for qid, _query, fut in pending:
            res_df = grouped.get(str(qid), pd.DataFrame(columns=batched_res.columns if isinstance(batched_res, pd.DataFrame) else []))
            try:
                entries = _postprocess_search_results(res_df, qid=qid)
                if not fut.done():
                    fut.set_result(entries)
            except Exception as exc:
                if not fut.done():
                    fut.set_exception(exc)

    async def flush(self) -> None:
        async with self._lock:
            await self._flush_locked()

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Retrieve documents relevant to answering the user's question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to retrieve relevant documents."
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

class GptOssAgent(pt.Transformer):
    def __init__(
        self,
        retriever: pt.Transformer,
        model_id: str,
        agent_k: int = 3,
        max_turns: int = 100,
        retriever_batch_size: int = 1,
        retriever_batch_wait_ms: int = 25,
    ):
        self.max_turns = max_turns
        self.retrieval_top_k = agent_k
        self.retriever_batch_size = max(1, int(retriever_batch_size))
        self.retriever_batch_wait_ms = max(0, int(retriever_batch_wait_ms))
        self.prompt = load_prompt(QA_PROMPT_PATH)

        self.retriever = retriever
        self.backend = _TerrierOpenAIBackend(model_id=model_id)

    @staticmethod
    def _assistant_tool_call_message(parsed: ParsedModelOutput) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": parsed.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in parsed.tool_calls
            ],
        }

    def _init_session(self, row: pd.Series) -> dict[str, Any]:
        qid = str(row["qid"])
        question = str(row["query"]).strip()
        if question and question[-1] != "?":
            question += "?"

        messages: list[dict[str, Any]] = [{"role": "system", "content": self.prompt}]
        messages.append({"role": "user", "content": f"Question: {question}"})

        return {
            "qid": qid,
            "query": question,
            "turn": 0,
            "done": False,
            "answer": "",
            "messages": messages,
            "all_queries": [],
            "all_docs": [],
            "all_docids": [],
            "all_ranked_lists": [],
            "all_errors": [],
            "all_thoughts": [],
            "all_input_tokens": [],
            "all_output_tokens": [],
        }

    def _finalize_session(self, session: dict[str, Any]) -> dict[str, Any]:
        return {
            "qid": session["qid"],
            "query": session["query"],
            "answer": session["answer"],
            "output": str(session["messages"]),
            "iteration": max(len(session["all_queries"]) - 1, 0),
            "all_queries": session["all_queries"],
            "all_docs": session["all_docs"],
            "all_docids": session["all_docids"],
            "all_ranked_lists": session["all_ranked_lists"],
            "all_errors": session["all_errors"],
            "all_thoughts": session["all_thoughts"],
            "all_input_tokens": session["all_input_tokens"],
            "all_output_tokens": session["all_output_tokens"],
        }

    @staticmethod
    def _record_error(session: dict[str, Any], turn: int, raw_query: str, sanitized_query: str, exc: Exception) -> None:
        session["all_errors"].append(
            {
                "qid": session["qid"],
                "iteration": turn,
                "qid_turn": f"{session['qid']}-{turn}",
                "raw_query": raw_query,
                "query": sanitized_query,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        )

    async def _search_async(self, search_batcher: _RetrieverBatcher | None, qid_turn: str, search_query: str) -> list[dict]:
        if search_batcher is None:
            return await asyncio.to_thread(
                _search,
                self.retriever,
                search_query,
                qid_turn,
            )
        return await search_batcher.submit(qid=qid_turn, query=search_query)

    async def _handle_parsed_response_async(self, session: dict[str, Any], parsed, search_batcher: _RetrieverBatcher | None) -> None:
        turn = int(session["turn"])
        usage = parsed.usage
        session["all_output_tokens"].append((turn, usage.completion_tokens or 0))
        session["all_input_tokens"].append((turn, usage.prompt_tokens or 0))
        if parsed.reasoning:
            session["all_thoughts"].append((turn, parsed.reasoning))

        if parsed.tool_calls:
            session["messages"].append(self._assistant_tool_call_message(parsed))
            for call in parsed.tool_calls:
                if call.name == "search":
                    raw_search_query = call.arguments.get("query", "")
                    search_query = raw_search_query.strip()  # Keep the backend self-contained; avoid adding a new data_utils dependency.
                    try:
                        full_doc_entries = await self._search_async(
                            search_batcher=search_batcher,
                            qid_turn=f"{session['qid']}-{turn}",
                            search_query=search_query,
                        )
                    except Exception as exc:
                        self._record_error(
                            session=session,
                            turn=turn,
                            raw_query=raw_search_query,
                            sanitized_query=search_query,
                            exc=exc,
                        )
                        session["done"] = True
                        return
                    agent_doc_entries = (
                        full_doc_entries[: self.retrieval_top_k]
                        if self.retrieval_top_k is not None
                        else full_doc_entries
                    )
                    retrieved_docids = [entry.get("docid") for entry in agent_doc_entries]
                    search_results = _format_docs(agent_doc_entries)
                    ranked_list = [
                        {
                            "qid_turn": f"{session['qid']}-{turn}",
                            "docid": entry.get("docid"),
                            "first_stage_rank": entry.get("first_stage_rank"),
                            "first_stage_score": entry.get("first_stage_score"),
                            "rank": entry.get("rank"),
                            "score": entry.get("score"),
                        }
                        for entry in full_doc_entries
                    ]
                    session["all_queries"].append((turn, search_query))
                    session["all_docs"].append((turn, search_results))
                    session["all_docids"].append((turn, retrieved_docids))
                    session["all_ranked_lists"].append((turn, ranked_list))
                    tool_content = (
                        f"query: {search_query}\n"
                        f"documents:\n{search_results}"
                    )
                else:
                    tool_content = f"Unsupported tool: {call.name}"

                session["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_content,
                    }
                )
            session["turn"] += 1
            return

        if parsed.content:
            session["answer"] = parsed.content
            session["messages"].append({"role": "assistant", "content": parsed.content})
            session["done"] = True
            return

        if parsed.finish_reason == "stop":
            session["done"] = True
            return

        session["done"] = True

    async def _run_sessions_async(self, sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        search_batcher = None
        if self.retriever_batch_size > 1:
            search_batcher = _RetrieverBatcher(
                search_pipeline=self.retriever,
                max_batch_size=self.retriever_batch_size,
                max_wait_ms=self.retriever_batch_wait_ms,
            )
        while True:
            active = [s for s in sessions if (not s["done"] and int(s["turn"]) < self.max_turns)]
            if not active:
                break

            responses = await asyncio.gather(
                *[self.backend.generate_async(messages=s["messages"]) for s in active],
                return_exceptions=True,
            )
            parsed_list = []
            for session, response in zip(active, responses):
                if isinstance(response, Exception):
                    self._record_error(
                        session=session,
                        turn=int(session["turn"]),
                        raw_query=session.get("query", ""),
                        sanitized_query=session.get("query", ""),
                        exc=response,
                    )
                    session["done"] = True
                    parsed_list.append(None)
                    continue
                try:
                    parsed_list.append(parse_chat_completion(response))
                except Exception as exc:
                    self._record_error(
                        session=session,
                        turn=int(session["turn"]),
                        raw_query=session.get("query", ""),
                        sanitized_query=session.get("query", ""),
                        exc=exc,
                    )
                    session["done"] = True
                    parsed_list.append(None)
            await asyncio.gather(
                *[
                    self._handle_parsed_response_async(session, parsed, search_batcher)
                    for session, parsed in zip(active, parsed_list)
                    if parsed is not None
                ]
            )

        if search_batcher is not None:
            await search_batcher.flush()

        for session in sessions:
            if not session["done"]:
                session["done"] = True

        return [self._finalize_session(s) for s in sessions]

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        if inp.empty:
            return pd.DataFrame(
                columns=[
                    "qid",
                    "query",
                    "answer",
                    "output",
                    "iteration",
                    "all_queries",
                    "all_docs",
                    "all_docids",
                    "all_ranked_lists",
                    "all_errors",
                    "all_thoughts",
                    "all_input_tokens",
                    "all_output_tokens",
                ]
            )
        sessions = [self._init_session(row) for _, row in inp.iterrows()]
        results = asyncio.run(self._run_sessions_async(sessions))
        return pd.DataFrame(results)
