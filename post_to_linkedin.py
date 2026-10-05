#!/usr/bin/env python3
"""
LinkedIn Automated Post Assistant via Playwright.
Uses persistent browser profile so you only need to log in once.
Prepares the post draft, attaches the demo video, and allows one-click review & submit.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

DEFAULT_POST_TEXT = """For our computer vision project at Universitas Multimedia Nusantara, my team and I built a real-time PPE compliance tracking system using YOLOv9-c.

Two main problems came up early on:
1. Labeling 22,000 images across 12 safety classes is expensive. We used semi-supervised learning with adaptive pseudo-labeling, training on 60% labeled data and 40% unlabelled images. That cut about 40% of the annotation work while still bumping mAP@0.5 to 83.68% (compared to 83.08% on the supervised baseline).
2. Standard IoU does not work well for safety gear. A person bounding box is huge compared to a hard hat or glasses, so IoU between a worker and a helmet is usually tiny even when the helmet is worn correctly. We wrote a spatial state machine using IoPPE (Intersection over PPE Area) combined with vertical anatomical zoning. Helmets are checked in the top 40% of the person box, vests in the torso region, and so on, running at 38.4 FPS.

We also ran EigenCAM saliency maps to check whether the model was actually attending to the gear or just memorizing background construction clutter.

The full repository with our inference script and paper is open source:
https://github.com/KennyUMN/ppe-compliance-detection

Built with Amanda Aliyah, Aisha Reissani Sopyan, Bryan Valleryan Alvonso, and Vaurolla Az Zahra."""

def main():
    parser = argparse.ArgumentParser(description="Post PPE Compliance project to LinkedIn via Playwright")
    parser.add_argument("--video", type=str, default="demo_linkedin_landscape.mp4", help="Video file to attach")
    parser.add_argument("--auto-publish", action="store_true", help="Automatically click the final Post button")
    parser.add_argument("--lang", choices=["en", "id"], default="en", help="Post language (en or id)")
    args = parser.parse_args()

    # Select text
    if args.lang == "id":
        post_text = """Untuk tugas akhir computer vision di Universitas Multimedia Nusantara, saya dan tim bikin sistem deteksi kepatuhan APD (Alat Pelindung Diri) real-time memakai YOLOv9-c.

Ada dua masalah teknis utama yang kami hadapi waktu pengerjaan:
1. Anotasi manual untuk 22.000 gambar di 12 kelas APD itu berat dan makan waktu. Kami pakai pendekatan semi-supervised learning dengan adaptive pseudo-labeling (60% data berlabel, 40% tanpa label). Hasilnya bisa menghemat sekitar 40% beban anotasi manual, dengan mAP@0.5 naik ke 83,68% dibanding baseline 83,08%.
2. Metrik IoU standar kurang cocok untuk mendeteksi pemakaian APD. Ukuran bounding box orang jauh lebih besar dibanding helm atau kacamata, jadi nilai IoU-nya selalu kecil meskipun helmnya terpasang rapi di kepala. Kami membuat state machine berbasis IoPPE (Intersection over PPE Area) dipadukan dengan pembagian zona anatomis tubuh. Deteksi helm dibatasi di 40% area atas tubuh pekerja, rompi di area dada/perut, dan seterusnya, jalan di kecepatan 38,4 FPS.

Kami juga memvalidasi representasi model menggunakan EigenCAM spatial saliency maps untuk memastikan deteksi fokus ke atribut pekerja, bukan ke background proyek.

Kode lengkap, script inferensi real-time, dan paper-nya sudah kami upload di GitHub:
https://github.com/KennyUMN/ppe-compliance-detection

Proyek ini dikerjakan bareng Amanda Aliyah, Aisha Reissani Sopyan, Bryan Valleryan Alvonso, dan Vaurolla Az Zahra."""
    else:
        post_text = DEFAULT_POST_TEXT

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        fallback = Path("/Users/kennyvws/ppe-compliance-detection/demo_linkedin_landscape.mp4")
        if fallback.exists():
            video_path = fallback
        else:
            print(f"[Warning] Video file {args.video} not found. Proceeding with text only.")
            video_path = None

    profile_dir = Path.home() / ".config" / "playwright_linkedin_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Launching Chromium with persistent profile:")
    print(f"Profile directory: {profile_dir}")
    print("=" * 65)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=False,
            viewport={"width": 1280, "height": 850},
            args=["--disable-blink-features=AutomationControlled"]
        )
        page = context.pages[0] if context.pages else context.new_page()

        print("\nNavigating to LinkedIn Feed...")
        page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded", timeout=45000)
        time.sleep(3)

        # Check if login is required
        if "login" in page.url or "checkpoint" in page.url or "signup" in page.url:
            print("\n" + "!" * 65)
            print("[ACTION REQUIRED] LinkedIn login required!")
            print("Please log in to your LinkedIn account in the browser window.")
            print("The script is monitoring and will continue as soon as you reach the Feed...")
            print("!" * 65 + "\n")

            # Poll for feed or login completion (wait up to 15 minutes)
            logged_in = False
            start_wait = time.time()
            while time.time() - start_wait < 900:  # 15 minutes
                curr_url = page.url
                if "feed" in curr_url or "/in/" in curr_url or "mynetwork" in curr_url:
                    logged_in = True
                    print("\n[SUCCESS] Detected login! Feed loaded successfully.")
                    time.sleep(3)
                    break
                time.sleep(2)

            if not logged_in:
                print(f"[Timeout] Did not detect login after 15 minutes. Current URL: {page.url}")
                context.close()
                return

        print("\n[OK] Currently on LinkedIn Feed.")
        time.sleep(2)

        # Click 'Start a post'
        print("Locating 'Start a post' trigger...")
        post_trigger_selectors = [
            "button:has-text('Start a post')",
            ".share-box-feed-entry__trigger",
            "button.artdeco-button:has-text('Start a post')",
            "div.share-box-feed-entry__top-bar",
            "button[aria-label*='Start a post']"
        ]

        post_button = None
        for sel in post_trigger_selectors:
            try:
                elem = page.wait_for_selector(sel, timeout=4000)
                if elem and elem.is_visible():
                    post_button = elem
                    print(f"Found trigger with selector: {sel}")
                    elem.click()
                    break
            except Exception:
                continue

        if not post_button:
            print("[Info] Attempting fallback click on share box area...")
            try:
                page.click(".share-box-feed-entry", timeout=3000)
            except Exception as e:
                print(f"[Warning] Could not trigger start post modal: {e}")

        time.sleep(2)

        # Handle video upload if available
        if video_path and video_path.exists():
            print(f"\nAttaching video: {video_path}...")
            # Look for file input or media button
            media_btn_selectors = [
                "button[aria-label='Add media']",
                "button[aria-label*='media']",
                "button[aria-label*='video']",
                "button:has-text('Media')"
            ]

            file_input = page.query_selector("input[type='file']")
            if not file_input:
                for sel in media_btn_selectors:
                    try:
                        btn = page.query_selector(sel)
                        if btn and btn.is_visible():
                            btn.click()
                            time.sleep(1)
                            file_input = page.query_selector("input[type='file']")
                            break
                    except Exception:
                        continue

            if file_input:
                file_input.set_input_files(str(video_path))
                print(f"[OK] Video file dispatched ({video_path.name}). Waiting for preview/processing...")
                time.sleep(4)

                # If LinkedIn shows a 'Next' button after media upload modal
                next_btn_selectors = [
                    "button.share-box-footer__primary-btn:has-text('Next')",
                    "button:has-text('Next')",
                    "button[aria-label='Next']"
                ]
                for n_sel in next_btn_selectors:
                    try:
                        n_btn = page.wait_for_selector(n_sel, timeout=3000)
                        if n_btn and n_btn.is_visible():
                            n_btn.click()
                            print("[OK] Clicked 'Next' in media modal.")
                            time.sleep(2)
                            break
                    except Exception:
                        continue

        # Fill text editor
        print("\nEntering post text into editor...")
        editor_selectors = [
            "div.ql-editor[contenteditable='true']",
            "div[role='textbox'][contenteditable='true']",
            ".editor-content div[contenteditable='true']",
            "div[data-placeholder*='talk about']"
        ]

        editor = None
        for sel in editor_selectors:
            try:
                elem = page.wait_for_selector(sel, timeout=5000)
                if elem and elem.is_visible():
                    editor = elem
                    print(f"Found editor with selector: {sel}")
                    break
            except Exception:
                continue

        if editor:
            editor.click()
            time.sleep(0.5)
            # Insert text line by line or paste
            editor.fill(post_text)
            print("[OK] Post text inserted successfully!")
        else:
            print("[Warning] Could not find text editor selector automatically.")
            print("Please paste the post text manually into the editor.")

        time.sleep(2)

        if args.auto_publish:
            print("\nAuto-publish enabled. Locating final 'Post' button...")
            post_submit_selectors = [
                "button.share-actions__primary-action:has-text('Post')",
                "button:has-text('Post'):not([disabled])",
                "button[aria-label='Post']"
            ]
            for sel in post_submit_selectors:
                try:
                    btn = page.wait_for_selector(sel, timeout=4000)
                    if btn and btn.is_visible() and btn.is_enabled():
                        btn.click()
                        print("[SUCCESS] Clicked 'Post'! Post is now publishing.")
                        time.sleep(5)
                        break
                except Exception:
                    continue
        else:
            print("\n" + "=" * 65)
            print("[READY FOR REVIEW]")
            print("The post draft and video have been prepared in your browser window.")
            print("Review the draft, and whenever you're ready, click 'Post' in the browser!")
            print("Press Ctrl+C in this terminal when you are done.")
            print("=" * 65)

            try:
                # Keep browser open for user to review and post
                while True:
                    time.sleep(2)
            except KeyboardInterrupt:
                print("\nClosing browser...")

        context.close()

if __name__ == "__main__":
    main()
