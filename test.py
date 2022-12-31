for root, dirs, files in os.walk("/data/NIA50/50-1/data/labeled/C"):
    if files.endswith("json"):
        print(os.path.join(root, files)
