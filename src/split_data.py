"""NEU-DET verisini train / val / test olarak böler..."""
from pathlib import Path
import pandas as pd 
from sklearn.model_selection import train_test_split
PROJECT_DIR=Path(__file__).resolve().parent.parent
IMG_DIR=PROJECT_DIR/"data"/"NEU-DET"/"IMAGES"
OUT_PATH=PROJECT_DIR/"splits"/"split.csv"
SEED=42
def class_from_name(Path):
    return Path.stem.rsplit("_",1)[0]
def main():
    images=sorted(IMG_DIR.glob("*.jpg"))
    df=pd.DataFrame({
        "file":[p.name for p in images],
        "class":[class_from_name(p) for p in images]

    })
    print("toplam resim sayısı:",len(df))
    #1.bolme:%70 train ,%30 gecici
    train_df,temp_df=train_test_split(
        df,test_size=0.30,stratify=df["class"],random_state=SEED
    )
    #geciciyi 2 ye ayır =>%15 val ,%15 test
    val_df,test_df=train_test_split(
        temp_df,test_size=0.50,stratify=temp_df["class"],random_state=SEED
    )
    train_df=train_df.assign(split="train")
    val_df=val_df.assign(split="val")
    test_df=test_df.assign(split="test")

    out=pd.concat([train_df,val_df,test_df])
    out=out.sort_values("file").reset_index(drop=True)
    OUT_PATH.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_PATH,index=False)
    print("KAYDEDİLDİ:",OUT_PATH)
    print(pd.crosstab(out["class"],out["split"]))
if __name__=="__main__":
        main()