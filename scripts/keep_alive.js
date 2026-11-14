// Keep the Streamlit Community Cloud dashboard awake.
//
//   node scripts/keep_alive.js [url] [screenshot.png]
//
// Free-tier apps sleep after a period without visitors and show a "this app has gone to sleep"
// page whose button must be clicked to wake it; a plain HTTP request does not count as a visit.
// This opens the app in a real headless browser, clicks the wake button if it is there, and
// waits for the dashboard's own title to render inside the app frame. Exits non-zero (and saves
// a screenshot) if the dashboard does not load, so a failed wake shows up as a failed run.
const { chromium } = require("playwright");

const URL = process.argv[2] || "https://aviation-llm-eval.streamlit.app/";
const SHOT = process.argv[3] || "keep_alive.png";
const TITLE = /Do LLMs agree with expert aviation safety coders/;
const WAKE = /get this app back up|wake (this )?app/i;

async function titleRendered(page) {
  for (const f of page.frames()) {
    try {
      const t = await f.locator("h1").first().innerText({ timeout: 1000 });
      if (TITLE.test(t)) return true;
    } catch (_) {}
  }
  return false;
}

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  let woke = false;
  try {
    await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 90_000 });
    const deadline = Date.now() + 180_000;
    while (Date.now() < deadline) {
      if (await titleRendered(page)) {
        console.log(`dashboard loaded${woke ? " after waking it" : " (already awake)"}`);
        await browser.close();
        return;
      }
      const wake = page.getByRole("button", { name: WAKE });
      if (!woke && (await wake.count()) > 0) {
        console.log("app is asleep: clicking the wake button");
        await wake.first().click();
        woke = true;
      }
      await page.waitForTimeout(5_000);
    }
    throw new Error("dashboard title did not render within 3 minutes");
  } catch (err) {
    await page.screenshot({ path: SHOT, fullPage: true }).catch(() => {});
    console.error(`keep-alive failed: ${err.message} (screenshot: ${SHOT})`);
    await browser.close();
    process.exit(1);
  }
})();
