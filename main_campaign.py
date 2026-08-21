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
            page.mouse.wheel(0, 400)
            
        elif action_id == 1:  # View Story
            stories = page.locator("canvas")
            if stories.count() > 0:
                stories.first.click()
                time.sleep(4)

        elif action_id == 2:  # Like Noise (لایک اولین پست در صفحه اصلی)
            like_btns = page.locator("svg[aria-label='Like']")
            if like_btns.count() > 0:
                like_btns.first.click()
                time.sleep(2)

        elif action_id == 3:  # Follow Noise (فالو کردن اکانت پیشنهادی)
            follow_btns = page.locator("button:has-text('Follow')")
            if follow_btns.count() > 0:
                follow_btns.first.click()
                time.sleep(2)

        elif action_id == 4:  # Like Target
            page.goto(f"https://www.instagram.com/{target_username}/", timeout=60000, wait_until="domcontentloaded")
            time.sleep(3)
            like_btns = page.locator("svg[aria-label='Like']")
            if like_btns.count() > 0:
                like_btns.first.click()
            account["is_target_liked"] = True

        elif action_id == 5:  # Follow Target
            page.goto(f"https://www.instagram.com/{target_username}/", timeout=60000, wait_until="domcontentloaded")
            time.sleep(3)
            follow_btn = page.locator("button:has-text('Follow')")
            if follow_btn.count() > 0:
                follow_btn.first.click()
            account["is_target_followed"] = True

    except Exception as e:
        print(f"خطا در اجرای اکشن {action_name}: {e}")
def run_campaign():
    target_username = "account_target_"
    accounts = load_accounts()

    try:
        with open("cookies.json", "r", encoding="utf-8") as f:
            all_cookies = json.load(f)
    except FileNotFoundError:
        print("ارور: فایل cookies.json پیدا نشد!")
        return

    CAMPAIGN_DAYS = 1
    ACTIVATION_PROBABILITY = 1.0 / CAMPAIGN_DAYS if CAMPAIGN_DAYS > 1 else 1.0

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

            if random.random() > ACTIVATION_PROBABILITY:
                print(f"اکانت {user} در این چرخه استراحت می‌کند.")
                continue

            print(f"شروع فعالیت اکانت: {user}")

            context = browser.new_context()
            context.add_cookies(all_cookies[user])
            page = context.new_page()

            try:
                page.goto("https://www.instagram.com", timeout=60000, wait_until="domcontentloaded")
                time.sleep(3)
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
