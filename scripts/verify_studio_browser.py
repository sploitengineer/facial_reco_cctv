"""Run explicit UI smoke checks against a running Studio (no camera or record changes)."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", default="tmp/studio_browser_qa")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width":1440,"height":1000}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("response", lambda response: errors.append(f"HTTP {response.status}: {response.url}") if response.status>=500 else None)
        page.goto(args.url)
        page.wait_for_function("() => document.querySelector('#engine-state').textContent.includes('Models ready')", timeout=90000)
        page.screenshot(path=str(output/"monitor-desktop.png"), full_page=True)
        for tab in ("enroll","profiles","events"):
            page.locator(f"[data-tab='{tab}']").click()
            page.wait_for_selector(f"#tab-{tab}:visible")
            page.screenshot(path=str(output/f"{tab}-desktop.png"), full_page=True)
            assert page.evaluate("() => document.documentElement.scrollWidth <= innerWidth"), f"Horizontal overflow: {tab}"
        page.locator("[data-tab='profiles']").click()
        page.locator("#profile-search").fill("zz_no_matching_profile")
        page.get_by_role("heading", name="No matching profiles").wait_for()
        page.locator("#profile-search").fill("")
        page.set_viewport_size({"width":390,"height":844})
        for tab in ("monitor","enroll","profiles","events"):
            page.locator(f"[data-tab='{tab}']").click()
            assert page.evaluate("() => document.documentElement.scrollWidth <= innerWidth"), f"Mobile overflow: {tab}"
        page.locator("[data-tab='monitor']").click()
        page.screenshot(path=str(output/"monitor-mobile.png"), full_page=True)
        browser.close()
    if errors:
        raise RuntimeError("Browser JavaScript errors: " + "; ".join(errors))
    print(json.dumps({"pages_checked":4,"sizes_checked":[1440,390],"javascript_errors":errors,"output":str(output)}))


if __name__ == "__main__":
    main()
