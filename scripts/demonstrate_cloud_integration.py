import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import app as webapp
from src import db

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PORT = 5000
BASE_URL = f"http://127.0.0.1:{PORT}"


def start_server():
    webapp.app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)


def run_demonstration():
    print("=" * 60)
    print("Starting VisionGuard Flask server in background thread...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(2)

    import requests
    try:
        health_resp = requests.get(f"{BASE_URL}/health", timeout=5)
        print("Flask Health Check:", health_resp.status_code, health_resp.json())
    except Exception as e:
        print("Failed to connect to Flask server:", e)
        sys.exit(1)

    print("Configuring Chrome headless browser...")
    chrome_opts = Options()
    chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--window-size=1280,950")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=chrome_opts)
    wait = WebDriverWait(driver, 30)

    try:
        # Step 1: Capture Homepage
        print("\n--- Capturing Homepage (Figure 8.1a) ---")
        driver.get(BASE_URL)
        time.sleep(1)
        home_path = FIGURES_DIR / "fig8_1a_flask_home.png"
        driver.save_screenshot(str(home_path))
        print(f"Saved: {home_path}")

        # Multiple Inputs to test
        test_inputs = [
            ("test_files/small_real.jpg", "fig8_2a_result_real1.png", "Real Artwork #1"),
            ("test_files/small_ai.jpg", "fig8_2b_result_ai1.png", "AI Art #1"),
            ("test_files/test_real_drawing.jpg", "fig8_2c_result_real2.png", "Real Artwork #2"),
            ("test_files/test_ai_portrait.jpg", "fig8_2d_result_ai2.png", "AI Art #2"),
        ]

        for img_rel_path, shot_name, label_desc in test_inputs:
            img_abs = os.path.abspath(img_rel_path)
            print(f"\n--- Testing Input: {label_desc} ({img_rel_path}) ---")
            driver.get(BASE_URL)
            file_input = wait.until(EC.presence_of_element_located((By.NAME, "image")))
            file_input.send_keys(img_abs)

            submit_btn = driver.find_element(By.CSS_SELECTOR, "button.btn")
            submit_btn.click()

            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "verdict")))
            time.sleep(1)

            verdict_el = driver.find_element(By.CLASS_NAME, "verdict")
            verdict_text = verdict_el.text
            print(f"Prediction verdict received from Cloud API: {verdict_text}")

            shot_path = FIGURES_DIR / shot_name
            driver.save_screenshot(str(shot_path))
            print(f"Saved screenshot: {shot_path}")

        # Step 3: Capture History Menu showing database logs
        print("\n--- Capturing History Menu (Figure 8.3a) ---")
        driver.get(f"{BASE_URL}/history")
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        time.sleep(1)
        history_path = FIGURES_DIR / "fig8_3a_history_menu.png"
        driver.save_screenshot(str(history_path))
        print(f"Saved: {history_path}")

        # Step 4: Show database records
        print("\n" + "=" * 60)
        print("DATABASE LOGS (VisionGuard SQLite/PostgreSQL Database)")
        print("=" * 60)
        rows = db.recent_predictions(10)
        print(f"{'ID':<4} | {'Timestamp (UTC)':<20} | {'Filename':<25} | {'Verdict':<14} | {'Conf':<7} | {'P(AI)':<7} | {'Model':<5} | {'Latency'}")
        print("-" * 105)
        for r in rows:
            print(f"{r.id:<4} | {str(r.created_at)[:19]:<20} | {r.filename:<25} | {r.label:<14} | {r.confidence:.2%} | {r.p_ai:<7.4f} | v{r.model_version:<4} | {r.latency_ms:.1f}ms")
        print("=" * 60)

    finally:
        driver.quit()
        print("Demonstration run completed successfully!")


if __name__ == "__main__":
    run_demonstration()
