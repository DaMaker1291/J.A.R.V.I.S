import asyncio
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any

from playwright.async_api import async_playwright


FRONTEND_URL = "http://localhost:5173/J.A.R.V.I.S/"


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-")
    return (slug[:60] or "command")


async def run_command_case(page, command: str, out_dir: Path, timeout_ms: int) -> Dict[str, Any]:
    ts = int(time.time())
    slug = _safe_slug(command)

    result: Dict[str, Any] = {
        "command": command,
        "slug": slug,
        "ok": False,
        "result_text": "",
        "screenshots": {},
    }

    # Ensure dashboard is ready
    await page.goto(FRONTEND_URL)
    await page.wait_for_load_state("networkidle")
    await page.wait_for_selector('input[placeholder*="boost productivity"]', timeout=15000)

    initial_path = out_dir / f"{ts}-{slug}-01-initial.png"
    await page.screenshot(path=str(initial_path), full_page=True)
    result["screenshots"]["initial"] = str(initial_path)

    command_input = page.locator('input[placeholder*="boost productivity"]').first
    await command_input.click()
    await command_input.fill(command)

    entered_path = out_dir / f"{ts}-{slug}-02-entered.png"
    await page.screenshot(path=str(entered_path), full_page=True)
    result["screenshots"]["entered"] = str(entered_path)

    execute_button = page.locator('button:has-text("Execute")').first
    await execute_button.click()

    # Wait for result card
    await page.wait_for_selector('text="Command Result"', timeout=timeout_ms)

    result_path = out_dir / f"{ts}-{slug}-03-result.png"
    await page.screenshot(path=str(result_path), full_page=True)
    result["screenshots"]["result"] = str(result_path)

    # Extract result text from the result card body
    result_text_locator = page.locator('h3:has-text("Command Result")').first.locator(
        'xpath=ancestor::*[self::div or self::section][1]'
    )

    # Fallback: the monospaced result box
    mono = page.locator('div.whitespace-pre-wrap').first
    if await mono.count() > 0:
        txt = await mono.inner_text()
    else:
        txt = await result_text_locator.inner_text()

    txt = (txt or "").strip()
    result["result_text"] = txt

    # Basic sanity checks
    if not txt:
        raise RuntimeError("Command Result rendered but text was empty")
    if "Failed to connect to backend" in txt or "Failed to connect" in txt:
        raise RuntimeError(f"Backend connection error shown in UI: {txt[:200]}")

    final_path = out_dir / f"{ts}-{slug}-04-final.png"
    await page.wait_for_timeout(750)
    await page.screenshot(path=str(final_path), full_page=True)
    result["screenshots"]["final"] = str(final_path)

    result["ok"] = True
    return result


async def run_multi_command_suite() -> int:
    out_dir = Path("screenshots") / "ui_multi_commands"
    out_dir.mkdir(parents=True, exist_ok=True)

    commands: List[Dict[str, Any]] = [
        {
            "command": "ping",
            "timeout_ms": 60_000,
        },
        {
            "command": "book a 7 day holiday to japan from NYC in april with a budget of $2500",
            "timeout_ms": 300_000,
        },
        {
            "command": "spider crawl https://example.com",
            "timeout_ms": 120_000,
        },
        {
            "command": "design a startup idea for an AI fitness coach and give me a go-to-market plan",
            "timeout_ms": 120_000,
        },
        {
            "command": "make a dropshipping business from scratch with no money: give step-by-step actionable plan",
            "timeout_ms": 120_000,
        },
        {
            "command": "research the best budget laptops under $500 and summarize top 5 options",
            "timeout_ms": 180_000,
        },
    ]

    results: List[Dict[str, Any]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()

        try:
            for i, item in enumerate(commands, start=1):
                cmd = item["command"]
                timeout_ms = int(item["timeout_ms"])
                print(f"\n=== CASE {i}/{len(commands)} ===")
                print(cmd)

                try:
                    case_result = await run_command_case(page, cmd, out_dir=out_dir, timeout_ms=timeout_ms)
                    results.append(case_result)
                    print("OK")
                    print(case_result["result_text"][:400])
                except Exception as e:
                    # Always capture error state screenshot
                    ts = int(time.time())
                    slug = _safe_slug(cmd)
                    err_path = out_dir / f"{ts}-{slug}-99-error.png"
                    await page.screenshot(path=str(err_path), full_page=True)
                    print(f"FAIL: {e}")
                    print(f"Error screenshot: {err_path}")
                    return 1

            # Save JSON summary
            import json

            summary_path = out_dir / "summary.json"
            with open(summary_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved summary: {summary_path}")
            return 0

        finally:
            await browser.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_multi_command_suite()))
