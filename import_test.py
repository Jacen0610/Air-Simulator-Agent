try:
    import stable_baselines3.common.policies
    # 打印出 policies 模块中所有可用的名称
    print("--- Contents of stable_baselines3.common.policies ---")
    print(sorted([name for name in dir(stable_baselines3.common.policies) if not name.startswith('_')]))

    # 额外检查：有时循环策略可能在算法自己的模块里
    import stable_baselines3.ppo.policies
    print("\n--- Contents of stable_baselines3.ppo.policies ---")
    print(sorted([name for name in dir(stable_baselines3.ppo.policies) if not name.startswith('_')]))

except ImportError as e:
    print(f"An import error occurred: {e}")
except Exception as e:
    print(f"An error occurred: {e}")