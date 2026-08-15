import os
import sys
import time
import json
import torch
import numpy as np
import datetime
from playwright.sync_api import sync_playwright

# ۱. تعریف ساختار شبکه عصبی (مطابق کدهای قبلی)
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim=5, action_dim=6):
        super(ActorCriticNetwork, self).__init__()
        
        # شبکه مشترک با ابعاد دقیق فایل ذخیره‌شده (128 -> 64)
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # خروجی اکشن‌ها (Actor)
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # خروجی ارزیابی (Critic)
        self.critic = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        shared_out = self.shared_net(x)
        return self.actor(shared_out), self.critic(shared_out)
    

# ۲. بارگذاری مدل PPO آموزش‌دیده
model = ActorCriticNetwork(input_dim=5, action_dim=6)
if os.path.exists("ppo_agent_model.pth"):
    model.load_state_dict(torch.load("ppo_agent_model.pth"))
    model.eval()
    print("✅ مدل PPO با موفقیت بارگذاری شد.")
else:
    print("⚠️ فایل ppo_agent_model.pth پیدا نشد!")

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

    # اعمال لایه شخصیت اکانت روی احتمالات
    probs[0] *= account["noise_affinity"]  # Scroll
    probs[1] *= account["noise_affinity"]  # View Story
    probs[2] *= account["noise_affinity"]  # Like Noise
    probs[3] *= account["noise_affinity"]  # Follow Noise
    probs[4] *= account["risk_tolerance"]  # Like Target
    probs[5] *= account["risk_tolerance"]  # Follow Target

    #  قانون مهم: اگر قبلاً تارگت فالو شده، احتمال اکشن 5 صفر شود!
    if account["is_target_followed"]:
        probs[5] = 0.0

    # نرمال‌سازی مجدد احتمالات
    if np.sum(probs) > 0:
        probs /= np.sum(probs)
    else:
        probs = np.ones(len(probs)) / len(probs)
    
    chosen_action = np.random.choice(len(probs), p=probs)
    
    # محاسبه تاخیر طبیعی بر اساس صبوری اکانت
    base_delay = np.random.uniform(3, 8)
    final_delay = base_delay * (1.5 - account["patience"])
    
    return chosen_action, final_delay


# ۵. اجرای اکشن روی مرورگر واقعی
def execute_action(page, action_id, target_username, account):
    action_name = action_names[action_id]
    print(f"🤖 تصمیم ایجنت برای {account['username']}: [{action_name}]")

    if action_id == 0:  # Scroll
        page.mouse.wheel(0, 400)
        
    elif action_id == 1:  # View Story
        print("👀 در حال دیدن استوری...")
        time.sleep(2)

    elif action_id == 2:  # Like Noise
        print("❤️ لایک کردن یک پست نویز...")
        time.sleep(1)

    elif action_id == 3:  # Follow Noise
        print("➕ فالو کردن یک اکانت پیشنهادی...")
        time.sleep(1)

    elif action_id == 4:  # Like Target
        print(f"🎯 رفتن به پیج target ({target_username}) و لایک...")
        page.goto(f"https://www.instagram.com/{target_username}/", timeout=60000)
        time.sleep(3)
        # 🟢 علامت‌گذاری انجام لایک target
        account["is_target_liked"] = True

    elif action_id == 5:  # Follow Target
        print(f"🎉 فالو کردن پیج target ({target_username})...")
        page.goto(f"https://www.instagram.com/{target_username}/", timeout=60000)
        time.sleep(3)
        # 🟢 علامت‌گذاری انجام فالو target
        account["is_target_followed"] = True


# 6. اجرای چرخه تست برای ۱۰ اکانت
def run_campaign():
    target_username = "instagram"  # آدرس پیج هدف
    accounts = load_accounts()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        
        for acc in accounts:
            if acc["is_target_followed"]:
                print(f"⏭️ اکانت {acc['username']} قبلاً پیج هدف را فالو کرده است. رد می‌شود.")
                continue
            print(f"\n==========================================")
            print(f"🚀 شروع فعالیت اکانت: {acc['username']}")
            print(f"==========================================")

            page = browser.new_page()
            page.goto("https://www.instagram.com", timeout=60000)
            time.sleep(3)


            # محاسبه تعداد گام‌ها بر اساس شخصیت اکانت
            # اکانت‌های با صبوری بالاتر، اکشن‌های بیشتری در یک نوبت انجام می‌دهند
            num_steps = int(3 + acc["patience"] * 3)  # نتیجه بین ۳ تا ۶ گام می‌شود

            for step in range(num_steps):
                dummy_obs = [1, 5.0, 2.1, 1.0, 14.5]
                action_id, delay = get_agent_decision(acc, dummy_obs)
    
                execute_action(page, action_id, target_username, acc)
                print(f"⏳ مکث انسانی: {delay:.1f} ثانیه...")
                time.sleep(delay)


            page.close()
            save_accounts(accounts)  # به روزرسانی دیتابیس

        browser.close()

if __name__ == "__main__":
    run_campaign()


