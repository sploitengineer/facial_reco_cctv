"""Explicit browser + real vision smoke test against an isolated Studio on port 8766."""

import json
from uuid import uuid4
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    output = Path("tmp/studio_browser_qa")
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    name = "UI enrollment check " + uuid4().hex[:6]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome",headless=True,args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            "--use-file-for-fake-video-capture=" + str(Path("tmp/studio_fake_camera.y4m").resolve())])
        context = browser.new_context(viewport={"width":1440,"height":1000},permissions=["camera"])
        existing = context.request.get("http://127.0.0.1:8766/api/state").json()
        assert all(p["name"].startswith("UI enrollment check") for p in existing["people"]), "Use an isolated QA registry"
        for person in existing["people"]:
            context.request.delete(f"http://127.0.0.1:8766/api/people/{person['id']}",
                headers={"X-SentryCare-Token":existing["token"]})
        page = context.new_page()
        page.on("pageerror",lambda error:errors.append(str(error)))
        page.on("response",lambda response:errors.append(f"HTTP {response.status}: {response.url}") if response.status>=500 else None)
        page.goto("http://127.0.0.1:8766")
        page.wait_for_function("() => document.querySelector('#engine-state').textContent.includes('Models ready')",timeout=90000)
        page.locator("[data-tab='enroll']").click()
        page.locator("#person-name").fill(name)
        page.get_by_role("button",name="Create profile",exact=True).click()
        page.locator("#enroll-person-info").get_by_text(name).wait_for()
        page.locator("#start-camera").click()
        page.wait_for_function("() => !document.querySelector('#capture-button').disabled",timeout=30000)
        for index,angle in enumerate(("front","left","right"),1):
            page.locator(f"[data-angle='{angle}']").click()
            page.locator("#capture-button").click()
            page.wait_for_function(f"() => document.querySelector('#sample-counter').textContent === '{index} photos'",timeout=15000)
        page.locator("#train-person").click()
        page.wait_for_function("() => document.querySelector('#enroll-person-info').textContent.includes('3 indexed')",timeout=30000)
        page.locator("[data-tab='monitor']").click()
        page.wait_for_function(f"() => document.querySelector('#live-people').textContent.includes('{name}')",timeout=90000)
        assert page.locator("#stat-faces").text_content()=="1"
        frame_status=page.locator("#latency").text_content()
        page.set_viewport_size({"width":390,"height":844})
        assert page.evaluate("() => document.documentElement.scrollWidth <= innerWidth"), "Connected mobile layout overflows"
        page.locator("#global-camera-stop").wait_for(state="visible")
        page.set_viewport_size({"width":1440,"height":1000})
        page.screenshot(path=str(output/"monitor-recognized.png"),full_page=True)
        page.locator("[data-tab='profiles']").click()
        page.get_by_role("heading",name=name).wait_for()
        profile = page.locator(".profile-card").filter(has=page.get_by_role("heading",name=name))
        assert profile.locator(".profile-images img").count()==3
        page.wait_for_function("() => [...document.querySelectorAll('.profile-images img')].every(image => image.complete && image.naturalWidth > 0)")
        profile.locator("[data-remove-person]").click()
        page.locator("#confirm-dialog").wait_for(state="visible")
        page.locator("#cancel-delete").click()
        page.get_by_role("heading",name=name).wait_for()
        page.locator("[data-tab='enroll']").click()
        page.locator("#global-camera-stop").click()
        assert page.locator("#video").is_hidden()
        data = context.request.get("http://127.0.0.1:8766/api/state").json()
        created = next(p for p in data["people"] if p["name"]==name)
        context.request.delete(f"http://127.0.0.1:8766/api/people/{created['id']}",
            headers={"X-SentryCare-Token":data["token"]})
        context.close()
        browser.close()
    if errors:
        raise RuntimeError("Browser JavaScript errors: "+"; ".join(errors))
    print(json.dumps({"profile_creation":True,"angle_capture":3,"real_sface_training":True,
                      "live_recognition":True,"photo_gallery":True,"delete_cancel":True,
                      "camera_shutdown":True,"processing":frame_status,"javascript_errors":errors}))


if __name__=="__main__":
    main()
