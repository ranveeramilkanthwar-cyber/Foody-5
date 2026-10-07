if __name__ == "__main__":
    import os, sys, json, time, warnings
    warnings.filterwarnings("ignore")
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"]       = "1"

    import torch
    import torchvision.transforms as T
    from torch import nn, optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
    from pathlib import Path
    from PIL import Image
    import pandas as pd
    from tqdm import tqdm
    from sklearn.metrics import classification_report

    BASE_DIR  = Path("M:/food-safety-ai")
    DATA_DIR  = BASE_DIR / "data"
    CKPT_DIR  = BASE_DIR / "checkpoints"
    MODEL_DIR = BASE_DIR / "models"
    for d in [DATA_DIR, CKPT_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*58)
    print("  FoodSafe AI  -  GPU Training  (Indian Food Regs)")
    print("="*58)

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA GPU! Exiting."); sys.exit(1)

    DEVICE = torch.device("cuda")
    print(f"\n[GPU] {torch.cuda.get_device_name(0)}  |  {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB  |  CUDA {torch.version.cuda}")

    CLASSES    = ["compliant","non_compliant","missing_fssai","expired","qr_present"]
    LABEL_MAP  = {c:i for i,c in enumerate(CLASSES)}
    IMG_SIZE   = 224
    BATCH_SIZE = 32
    EPOCHS     = 5
    LR         = 3e-4

    # --- Dataset ---
    print(f"\n[1/3] Loading dataset...")
    csv_path = DATA_DIR / "dataset_index.csv"
    if not csv_path.exists():
        print("  Generating dataset first...")
        import subprocess
        subprocess.run([sys.executable, str(BASE_DIR/"generate_dataset.py")], check=True)

    df = pd.read_csv(csv_path)
    df = df[df["path"].apply(lambda x: Path(x).exists())].reset_index(drop=True)
    if len(df) == 0:
        print("  No images! Run generate_dataset.py first."); sys.exit(1)

    print(f"  {len(df)} images  |  {df['label'].value_counts().to_dict()}")
    if "category" in df.columns:
        print(f"  {df['category'].nunique()} Indian food categories")

    TRAIN_TF = T.Compose([
        T.Resize((IMG_SIZE,IMG_SIZE)), T.RandomHorizontalFlip(),
        T.RandomRotation(15), T.ColorJitter(.3,.3,.2,.05),
        T.RandomAffine(0,translate=(.1,.1),scale=(.9,1.1)),
        T.RandomPerspective(.15,.3), T.ToTensor(),
        T.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])
    VAL_TF = T.Compose([
        T.Resize((IMG_SIZE,IMG_SIZE)), T.ToTensor(),
        T.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])

    class FoodDS(Dataset):
        def __init__(self, df, tf): self.df=df.reset_index(drop=True); self.tf=tf
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            r = self.df.iloc[i]
            try:    img = Image.open(r["path"]).convert("RGB")
            except: img = Image.new("RGB",(IMG_SIZE,IMG_SIZE),(200,200,200))
            return self.tf(img), int(r["label_id"])

    train_df = df.groupby("label",group_keys=False).apply(lambda x: x.iloc[:int(len(x)*.85)])
    val_df   = df.groupby("label",group_keys=False).apply(lambda x: x.iloc[int(len(x)*.85):])
    # num_workers=0 required on Windows
    train_ld = DataLoader(FoodDS(train_df,TRAIN_TF), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_ld   = DataLoader(FoodDS(val_df,  VAL_TF),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    print(f"  Train: {len(train_df)}  |  Val: {len(val_df)}")

    # --- Model ---
    print(f"\n[2/3] Building EfficientNet-B4...")
    base = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    for p in base.parameters(): p.requires_grad = False
    inf = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(p=.4, inplace=True),
        nn.Linear(inf, 512),
        nn.LayerNorm(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=.3),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, len(CLASSES)),
    )
    model = base.to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Model on {DEVICE}  |  Trainable: {trainable:,} / {total:,}")

    # GPU sanity check (batch=2, eval mode)
    model.eval()
    with torch.no_grad():
        _t = torch.zeros(2,3,IMG_SIZE,IMG_SIZE).to(DEVICE)
        _o = model(_t)
    print(f"  GPU forward-pass OK  |  output: {_o.shape}")
    print(f"  VRAM used: {torch.cuda.memory_allocated()/1e6:.0f} MB")
    model.train()

    # --- Train ---
    print(f"\n[3/3] Training {EPOCHS} epochs on {torch.cuda.get_device_name(0)}...\n")
    criterion = nn.CrossEntropyLoss(label_smoothing=.1)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR,
                    steps_per_epoch=len(train_ld), epochs=EPOCHS)
    scaler   = torch.cuda.amp.GradScaler()
    history  = []
    best_acc = 0.0
    last_p   = []
    last_l   = []

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        model.train()
        tr_loss=tr_cor=0
        bar = tqdm(train_ld, desc=f"  Ep{epoch}/{EPOCHS} TRAIN", ncols=80)
        for imgs,lbls in bar:
            imgs=imgs.to(DEVICE,non_blocking=True)
            lbls=lbls.to(DEVICE,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out=model(imgs); loss=criterion(out,lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); scheduler.step()
            tr_loss+=loss.item()*imgs.size(0)
            tr_cor +=(out.argmax(1)==lbls).sum().item()
            bar.set_postfix(loss=f"{loss.item():.3f}", gpu=f"{torch.cuda.memory_allocated()/1e6:.0f}MB")

        tr_loss/=len(train_df); tr_acc=tr_cor/len(train_df)

        model.eval()
        va_loss=va_cor=0; all_p=[]; all_l=[]
        with torch.no_grad():
            for imgs,lbls in tqdm(val_ld, desc=f"  Ep{epoch}/{EPOCHS} VAL  ", ncols=80):
                imgs=imgs.to(DEVICE,non_blocking=True)
                lbls=lbls.to(DEVICE,non_blocking=True)
                with torch.cuda.amp.autocast():
                    out=model(imgs); loss=criterion(out,lbls)
                va_loss+=loss.item()*imgs.size(0)
                va_cor +=(out.argmax(1)==lbls).sum().item()
                all_p.extend(out.argmax(1).cpu().numpy())
                all_l.extend(lbls.cpu().numpy())

        va_loss/=len(val_df); va_acc=va_cor/len(val_df)
        elapsed=time.time()-t0; last_p=all_p; last_l=all_l

        bar_s = "#"*int(va_acc*25)+"-"*(25-int(va_acc*25))
        print(f"\n  >> Ep{epoch:02d}/{EPOCHS}  train={tr_acc:.4f}({tr_loss:.4f})  val={va_acc:.4f}[{bar_s}]  {elapsed:.0f}s  gpu={torch.cuda.memory_allocated()/1e6:.0f}MB\n")

        torch.save({"epoch":epoch,"model_state":model.state_dict(),"val_acc":va_acc,"classes":CLASSES},
                   CKPT_DIR/f"ep{epoch:02d}_acc{va_acc:.4f}.pt")
        if va_acc > best_acc:
            best_acc=va_acc
            torch.save(model.state_dict(), MODEL_DIR/"best_model.pt")
            print(f"  *** BEST MODEL saved -> M:/food-safety-ai/models/best_model.pt  (acc={va_acc:.4f}) ***\n")

        history.append({"epoch":epoch,"train_loss":round(tr_loss,4),"train_acc":round(tr_acc,4),
                        "val_loss":round(va_loss,4),"val_acc":round(va_acc,4),"time_s":round(elapsed,1)})

    with open(DATA_DIR/"training_history.json","w") as f: json.dump(history,f,indent=2)
    with open(MODEL_DIR/"config.json","w") as f:
        json.dump({"classes":CLASSES,"img_size":IMG_SIZE,"num_classes":len(CLASSES),"model":"efficientnet_b4"},f,indent=2)

    print("="*58)
    print(f"  TRAINING COMPLETE!")
    print(f"  Best val accuracy : {best_acc:.4f}  ({best_acc*100:.2f}%)")
    print(f"  Model  : M:/food-safety-ai/models/best_model.pt")
    print(f"  Server : python M:/food-safety-ai/server.py")
    print("="*58+"\n")
    print(classification_report(last_l,last_p,target_names=CLASSES,zero_division=0))
