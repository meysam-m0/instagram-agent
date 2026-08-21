import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from playwright.sync_api import sync_playwright

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim=5, action_dim=6):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        shared_out = self.shared_net(x)
        return self.actor(shared_out), self.critic(shared_out)

model = ActorCriticNetwork()
if os.path.exists("ppo_agent_model.pth"):
    model.load_state_dict(torch.load("ppo_agent_model.pth"), strict=False)
    model.eval()

action_names = {
    0: "Scroll",
    1: "View Story",
    2: "Like Noise",
    3: "Follow Noise",
    4: "Like Target",
    5: "Follow Target"
}

def load_accounts():
    with open("accounts.json", "r", encoding="utf-8") as f:
        return json.load(f)

def save_accounts(accounts):
    with open("accounts.json", "w", encoding="utf-8") as f:
        json.dump(accounts, f, indent=4, ensure_ascii=False)

def get_agent_decision(account, current_obs):
    obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0)
    with torch.no_grad():
        action_probs, _ = model(obs_tensor)
        probs = action_probs.squeeze().numpy()

    probs[0] *= account.get("noise_affinity", 1.0)
    probs[1] *= account.get("noise_affinity", 1.0)
    probs[2] *= account.get("noise_affinity", 1.0)
    probs[3] *= account.get("noise_affinity", 1.0)
    probs[4] *= account.get("risk_tolerance", 1.0)
    probs[5] *= account.get("risk_tolerance", 1.0)

    if account.get("is_target_followed", False):
        probs[5] = 0.0

    if np.sum(probs) > 0:
        probs /= np.sum(probs)
    else:
        probs = np.ones(len(probs)) / len(probs)
    
    chosen_action = np.random.choice(len(probs), p=probs)
    base_delay = np.random.uniform(3, 8)
    final_delay = base_delay * (1.5 - account.get("patience", 0.5))
    
    return chosen_action, final_delay

def execute_action(page, action_id, target_username, account):
    action_name = action_names[action_id]
    print(f"🤖 تصمیم ایجنت برای {account['username']}: [{action_name}]")

    try:
        if action_id == 0:  # Scroll
            page.mouse.wheel(0, 500)
            time.sleep(2)
            
        elif action_id == 1:  # View Story
            story = page.locator("canvas, div[role='button']:has-text('Story')").first
            if story.is_visible():
                story.click()
                time.sleep(5)

        elif action_id == 2:  # Like Noise
            # کلیک روی اولین آیکون لایک موجود در صفحه
            like_btn = page.locator("span:has(svg[aria-label='Like']), span:has(svg[aria-label='پسندیدن'])").first
            if like_btn.is_visible():
                like_btn.click()
                print("✅ پست نویز لایک شد.")
                time.sleep(2)

        elif action_id == 3:  # Follow Noise
            # کلیک روی دکمه‌های فالو پیشنهادی
            follow_btn = page.locator("button:has-text('Follow'), button:has-text('دنبال کردن')").first
            if follow_btn.is_visible():
                follow_btn.click()
                print("✅ یک اکانت نویز فالو شد.")
                time.sleep(2)

        elif action_id == 4:  # Like Target
            page.goto(f"https://www.instagram.com/{target_username}/", timeout=60000, wait_until="networkidle")
            time.sleep(3)
            # باز کردن اولین پست
            first_post = page.locator("article a[href*='/p/']").first
            if first_post.is_visible():
                first_post.click()
                time.sleep(3)
                like_btn = page.locator("span:has(svg[aria-label='Like']), span:has(svg[aria-label='پسندیدن'])").first
                if like_btn.is_visible():
                    like_btn.click()
                    print("✅ پست پیج هدف لایک شد.")
                    account["is_target_liked"] = True
         elif action_id == 5:  # Follow Target
            target_url = f"https://www.instagram.com/{target_username}/"
            page.goto(target_url, timeout=60000, wait_until="domcontentloaded")
            time.sleep(4)
            
            # بررسی اگر به لاگین منتقل شد، همان‌جا لاگین کند
            if "login" in page.url:
                print(f"🔑 کوکی منقضی شده! در حال لاگین خودکار برای {user}...")
                pwd = account.get("password", "")
                if pwd:
                    page.fill("input[name='username']", user)
                    page.fill("input[name='password']", pwd)
                    page.click("button[type='submit']")
                    time.sleep(8)
                    # بعد از لاگین دوباره به پیج هدف برود
                    page.goto(target_url, timeout=60000, wait_until="domcontentloaded")
                    time.sleep(4)
                else:
                    print(f"❌ رمز عبور برای {user} در accounts.json یافت نشد!")
                    return

            # کلیک روی دکمه فالو
            follow_btn = page.locator("header button").filter(has_text=["Follow", "دنبال کردن", "فالو"]).first
            if follow_btn.is_visible():
                follow_btn.scroll_into_view_if_needed()
                follow_btn.click()
                print("✅ پیج هدف با موفقیت فالو شد.")
                account["is_target_followed"] = True
            else:
                print("❌ دکمه فالو پیدا نشد.")

    except Exception as e:
        print(f"⚠️ خطا در اجرای اکشن {action_name}: {e}")


def check_and_login(page, context, user, password, all_cookies):
    if "login" in page.url:
        print(f"🔑 کوکی منقضی شده! در حال لاگین خودکار برای {user}...")
        page.fill("input[name='username']", user)
        page.fill("input[name='password']", password)
        page.click("button[type='submit']")
        time.sleep(8)
        
        if "login" not in page.url:
            print(f"✅ لاگین جدید برای {user} موفقیت‌آمیز بود.")
            # به‌روزرسانی کوکی‌های جدید در فایل
            all_cookies[user] = context.cookies()
            with open("cookies.json", "w", encoding="utf-8") as f:
                json.dump(all_cookies, f, indent=4)
            return True
        else:
            print(f"❌ لاگین ناموفق برای {user}")
            return False
    return True


def run_campaign():
    target_username = "account_target_"  # نام کاربری پیج هدف را اینجا وارد کنید
    accounts = load_accounts()

    try:
        with open("cookies.json", "r", encoding="utf-8") as f:
            all_cookies = json.load(f)
    except FileNotFoundError:
        print("ارور: فایل cookies.json پیدا نشد!")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for acc in accounts:
            user = acc["username"]

            if user not in all_cookies:
                print(f"فایل کوکی برای {user} پیدا نشد!")
                continue

            if acc.get("is_target_followed", False):
                print(f"اکانت {user} قبلاً پیج هدف را فالو کرده است. رد می‌شود.")
                continue

            print(f"شروع فعالیت اکانت: {user}")

            context = browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            context.add_cookies(all_cookies[user])
            page = context.new_page()

            try:
                page.goto("https://www.instagram.com", timeout=60000, wait_until="domcontentloaded")
                time.sleep(4)
            except Exception as e:
                print(f"خطا در باز کردن صفحه برای {user}: {e}")
                page.close()
                context.close()
                continue

            num_steps = int(3 + acc.get("patience", 0.5) * 3)

            for step in range(num_steps):
                dummy_obs = [1, 5.0, 2.1, 1.0, 14.5]
                action_id, delay = get_agent_decision(acc, dummy_obs)
                
                execute_action(page, action_id, target_username, acc)
                print(f"⏳ مکث انسانی: {delay:.1f} ثانیه...")
                time.sleep(delay)

            page.close()
            context.close()
            save_accounts(accounts)

        browser.close()

if __name__ == "__main__":
    run_campaign()
