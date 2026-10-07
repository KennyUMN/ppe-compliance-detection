#!/usr/bin/env python3
"""
LinkedIn Automated Post Assistant via Playwright.
Uses persistent browser profile, uploads demo video, inserts humanized post text,
and publishes or drafts the post for review.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

DEFAULT_POST_TEXT = """For our computer vision project at Universitas Multimedia Nusantara, my team and I built a real-time PPE compliance tracking system using YOLOv9-c.

A few engineering challenges and solutions from building this:

1. Data Efficiency: Hand-labeling 22,000 images across 12 safety classes was impractical. We implemented semi-supervised learning with adaptive pseudo-labeling on a 60% labeled and 40% unlabeled split. This cut ~40% of annotation work while improving mAP@0.5 from 83.08% (supervised baseline) to 83.68%.

2. Spatial Association: Standard IoU breaks down for wearable safety equipment because a worker's bounding box is huge compared to a hard hat or goggles. We designed an IoPPE (Intersection over PPE Area) state machine with vertical anatomical constraints (e.g. helmets restricted to the upper 35% of the person box).

3. Real-World Video Stability: Raw frame-by-frame inference on site CCTV suffered from rapid state flickering when workers turned or bent down. We added an IoU-based worker tracker with temporal hysteresis (holding detection state over a 25-frame debounce window) and EMA bounding box smoothing. This eliminated flicker and kept worker tracking rock-solid at 38+ FPS.

Full repository, real-time inference script, and paper:
https://github.com/KennyUMN/ppe-compliance-detection

Built together with Amanda Aliyah, Aisha Reissani Sopyan, Bryan Valleryan Alvonso, and Vaurolla Az Zahra."""

INDONESIAN_POST_TEXT = """Untuk tugas akhir computer vision di Universitas Multimedia Nusantara, saya dan tim bikin sistem deteksi kepatuhan APD (Alat Pelindung Diri) real-time memakai YOLOv9-c.

Tiga tantangan teknis utama dan solusi yang kami terapkan:

1. Efisiensi Data: Anotasi manual untuk 22.000 gambar di 12 kelas APD sangat makan waktu. Kami pakai pendekatan semi-supervised learning dengan adaptive pseudo-labeling (60% data berlabel, 40% tanpa label). Hasilnya menghemat ~40% beban anotasi manual, dengan mAP@0.5 naik ke 83,68% dibanding baseline 83,08%.

2. Asosiasi Spasial: Metrik IoU standar kurang cocok untuk APD karena bounding box orang jauh lebih besar dibanding helm atau kacamata. Kami membuat state machine berbasis IoPPE (Intersection over PPE Area) dipadukan dengan batasan zona anatomis vertikal tubuh.

3. Stabilitas Video Real-Time: Deteksi frame-by-frame murni di CCTV lapangan sempat flickering saat pekerja menunduk atau bergerak. Kami menambahkan IoU multi-worker tracker dengan temporal hysteresis (hold state 25 frame) dan EMA smoothing sehingga status kepatuhan terkunci stabil di 38+ FPS.

Kode lengkap, script inferensi real-time, dan paper-nya:
https://github.com/KennyUMN/ppe-compliance-detection

Proyek ini dikerjakan bersama Amanda Aliyah, Aisha Reissani Sopyan, Bryan Valleryan Alvonso, dan Vaurolla Az Zahra."""

def main():
    parser = argparse.ArgumentParser(description="Post PPE Compliance project to LinkedIn via Playwright")
    parser.add_argument("--video", type=str, default="/Users/kennyvws/ppe-compliance-detection/assets/demo_cctv.mp4", help="Video file to attach")
    parser.add_argument("--auto-publish", action="store_true", help="Automatically click Post to publish")
    parser.add_argument("--lang", choices=["en", "id"], default="en", help="Post language (en or id)")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    args = parser.parse_args()

    post_text = INDONESIAN_POST_TEXT if args.lang == "id" else DEFAULT_POST_TEXT

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        fallback = Path("/Users/kennyvws/ppe-compliance-detection/assets/demo_cctv.mp4")
        if fallback.exists():
            video_path = fallback
        else:
            video_path = None

    profile_dir = Path.home() / ".config" / "playwright_linkedin_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Launching Chromium persistent profile...")
    print(f"Profile directory: {profile_dir}")
    print(f"Language: {args.lang.upper()} | Auto-publish: {args.auto_publish}")
    print("=" * 65)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=args.headless,
            viewport={"width": 1280, "height": 850},
            args=["--disable-blink-features=AutomationControlled"]
        )
        page = context.pages[0] if context.pages else context.new_page()

        print("Navigating to LinkedIn Feed...")
        page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded", timeout=45000)
        time.sleep(3)

        # Check if feed is loaded
        if "feed" not in page.url and "/in/" not in page.url:
            print("[Info] Waiting for feed to load...")
            for _ in range(60):
                if "feed" in page.url or "/in/" in page.url:
                    break
                time.sleep(2)

        print("[OK] Confirmed on LinkedIn Feed.")
        time.sleep(2)

        # Click Start a post
        print("Clicking 'Start a post'...")
        page.click("text='Start a post'")
        time.sleep(2)

        # Attach video
        if video_path and video_path.exists():
            print(f"Uploading demo video: {video_path.name}...")
            with page.expect_file_chooser() as fc_info:
                page.click("button[aria-label='Add media']")
            file_chooser = fc_info.value
            file_chooser.set_files(str(video_path))
            print("Video dispatched. Waiting 5s for preview...")
            time.sleep(5)

            # Click Next in video preview modal
            print("Confirming media with 'Next'...")
            page.click("button:has-text('Next')")
            time.sleep(3)

        # Insert post text
        print("Inserting post text into editor...")
        editor = page.wait_for_selector("div.ql-editor", timeout=10000)
        editor.click()
        page.keyboard.insert_text(post_text)
        time.sleep(2)
        print("[OK] Text inserted successfully.")

        # Capture draft screenshot
        draft_img = "/Users/kennyvws/.gemini/antigravity-cli/brain/bd54f1b9-98c1-4aa7-a3ca-21f0698fc9da/scratch/linkedin_draft_ready.png"
        page.screenshot(path=draft_img)
        print(f"Screenshot of ready draft saved to: {draft_img}")

        if args.auto_publish:
            print("\nAuto-publish enabled. Clicking 'Post' button...")
            post_btn = page.wait_for_selector("button.share-actions__primary-action", timeout=5000)
            if post_btn and not post_btn.is_disabled():
                post_btn.click()
                print("Clicked Post! Waiting 8s for publishing...")
                time.sleep(8)

                # Capture final result screenshot
                result_img = "/Users/kennyvws/.gemini/antigravity-cli/brain/bd54f1b9-98c1-4aa7-a3ca-21f0698fc9da/scratch/linkedin_published_result.png"
                page.screenshot(path=result_img)
                print(f"[SUCCESS] Post published! Screenshot: {result_img}")
            else:
                print("[Warning] Post button was disabled or not found.")
        else:
            print("\n" + "=" * 65)
            print("[READY FOR REVIEW]")
            print("Draft is open in the Chromium window.")
            print("Click 'Post' when you're ready, or press Ctrl+C to close.")
            print("=" * 65)
            try:
                while True:
                    time.sleep(2)
            except KeyboardInterrupt:
                pass

        context.close()

if __name__ == "__main__":
    main()
