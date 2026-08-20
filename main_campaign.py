import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from playwright.sync_api import sync_playwright

# ۱. تعریف ساختار شبکه عصبی PPO
class ActorCriticNetwork(nn.Module):
    def init(self, input_dim=5, action_dim=6):
        super(ActorCriticNetwork, self).__init__()
        
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

# ۲. بارگذاری مدل PPO
model = ActorCriticNetwork(state_dim=5, action_dim=6)
if os.path.exists("ppo_agent_model.pth"):
    model.load_state_dict(torch.load("ppo_agent_model.pth"))
    model.eval()
else:
    print(" فایل ppo_agent_model.pth پیدا نشد!")

action_names = {
    0: "Scroll",
    1: "View Story",
    2: "Like Noise",
    3: "Follow Noise",
    4: "Like Target",
    5: "Follow Target"
}

# ۳. توابع مدیریت دیتابیس اکانت‌ها
def load_accounts():
    with open("accounts.json", "r", encoding="utf-8") as f:
        return json.load(f)

def save_accounts(accounts):
    with open("accounts.json", "w", encoding="utf-8") as f:
        json.dump(accounts, f, indent=4, ensure_ascii=False)

# ۴. تصمیم‌گیری ایجنت بر اساس ویژگی‌های هر اکانت
def get_agent_decision(account, current_obs):
    obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0)
    with torch.no_grad():
        action_probs, _ = model(obs_tensor)
        probs = action_probs.squeeze().numpy()

    # اعمال ویژگی‌های شخصیتی اکانت
    probs[0] *= account["noise_affinity"]  # Scroll
    probs[1] *= account["noise_affinity"]  # View Story
    probs[2] *= account["noise_affinity"]  # Like Noise
    probs[3] *= account["noise_affinity"]  # Follow Noise
    probs[4] *= account["risk_tolerance"]  # Like Target
    probs[5] *= account["risk_tolerance"]  # Follow Target

    # اگر قبلاً فالو شده، احتمال صفر شود
    if account.get("is_target_followed", False):
        probs[5] = 0.0

    # نرمال‌سازی مجدد
    if np.sum(probs) > 0:
        probs /= np.sum(probs)
    else:
        probs = np.ones(len(probs)) / len(probs)
    
    chosen_action = np.random.choice(len(probs), p=probs)
    
    # محاسبه تاخیر طبیعی
    base_delay = np.random.uniform(3, 8)
    final_delay = base_delay * (1.5 - account["patience"])
    
    return chosen_action, final_delay

# ۵. ورود واقعی به اکانت (Login)
def login_to_instagram(page, username, password):
    print(f" در حال ورود به اکانت: {username}...")
    page.goto("https://www.instagram.com/accounts/login/", timeout=60000)
    time.sleep(4)
    
    try:
        page.fill("input[name='username']", username)
        page.fill("input[name='password']", password)
        page.click("button[type='submit']")
        time.sleep(7)  # صبر برای انجام فرایند لاگین
    except Exception as e:
        print(f" خطا در فرم لاگین {username}: {e}")

# ۶. اجرای اکشن روی مرورگر
def execute_action(page, action_id, target_username, account):
    action_name = action_names[action_id]
    print(f"🤖 تصمیم ایجنت برای {account['username']}: [{action_name}]")

    if action_id == 0:  # Scroll
        page.mouse.wheel(0, 400)
        
    elif action_id == 1:  # View Story
        print(" در حال دیدن استوری...")
        time.sleep(2)

    elif action_id == 2:  # Like Noise
        print(" لایک کردن یک پست نویز...")
        time.sleep(1)

    elif action_id == 3:  # Follow Noise
        print(" فالو کردن یک اکانت پیشنهادی...")
        time.sleep(1)
    elif action_id == 4:  # Like Target
        print(f" رفتن به پیج target ({target_username}) و لایک...")
        page.goto(f"https://www.instagram.com/{target_username}/", timeout=60000)
        time.sleep(3)
        account["is_target_liked"] = True

    elif action_id == 5:  # Follow Target
        print(f" فالو کردن پیج target ({target_username})...")
        page.goto(f"https://www.instagram.com/{target_username}/", timeout=60000)
        time.sleep(3)
        account["is_target_followed"] = True

# ۷. اجرای چرخه کمپین
def run_campaign():
    target_username = "account_target_"  # آدرس پیج هدف
    accounts = load_accounts()

    CAMPAIGN_DAYS = 1  # تنظیم بازه زمانی کمپین (به روز)
    ACTIVATION_PROBABILITY = 1.0 / CAMPAIGN_DAYS if CAMPAIGN_DAYS > 1 else 1.0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

        for acc in accounts:
            if acc.get("is_target_followed", False):
                print(f"اکانت {acc['username']} قبلاً پیج هدف را فالو کرده است. رد می‌شود.")
                continue

            # تصمیم‌گیری تصادفی برای استراحت یا فعالیت روزانه
            if random.random() > ACTIVATION_PROBABILITY:
                print(f" اکانت {acc['username']} در این چرخه استراحت می‌کند.")
                continue

            print(f" شروع فعالیت اکانت: {acc['username']}")

            page = browser.new_page()
            
            # دریافت پسورد (با پشتیبانی از هر دو شکل املایی pasword و password)
            acc_password = acc.get("password") or acc.get("pasword")
            
            # لاگین با اطلاعات اختصاصی همین اکانت
            login_to_instagram(page, acc["username"], acc_password)

            # محاسبه تعداد گام‌ها
            num_steps = int(3 + acc["patience"] * 3)

            for step in range(num_steps):
                dummy_obs = [1, 5.0, 2.1, 1.0, 14.5]
                action_id, delay = get_agent_decision(acc, dummy_obs)
    
                execute_action(page, action_id, target_username, acc)
                print(f"⏳ مکث انسانی: {delay:.1f} ثانیه...")
                time.sleep(delay)

            page.close()
            save_accounts(accounts)  # ذخیره وضعیت جدید اکانت در دیتابیس

        browser.close()

if __name__ == "__main__":
    run_campaign()
