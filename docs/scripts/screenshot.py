"""Screenshot all HTML design files at their updated dimensions."""
import os
from playwright.sync_api import sync_playwright

HTML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")

files = [
    ("hook-opening.html", 900, 560),
    ("pinyin-map.html", 900, 650),
    ("go-kan-contrast.html", 900, 620),
    ("phonetic-family.html", 900, 640),
    ("data-chart.html", 900, 940),
    ("iron-laws.html", 900, 900),
    ("readme-card.html", 900, 680),
]

with sync_playwright() as p:
    browser = p.chromium.launch()
    for fname, w, h in files:
        path = os.path.join(HTML_DIR, fname)
        if not os.path.exists(path):
            print(f"SKIP (not found): {fname}")
            continue
        page = browser.new_page(viewport={"width": w, "height": h})
        page.goto(f"file://{path}", wait_until="networkidle")
        page.set_viewport_size({"width": w, "height": h})
        png_name = fname.replace(".html", ".png")
        png_path = os.path.join(HTML_DIR, png_name)
        page.screenshot(path=png_path, full_page=False)
        print(f"OK: {png_name} ({w}x{h})")
        page.close()
    browser.close()
print("Done.")
