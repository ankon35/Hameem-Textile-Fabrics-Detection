# FabricEye QC: AI fabric inspection MVP

A software demo for the Hameem Textile AI Engineer interview. It shows how cameras and AI on the
fabric inspection machine can catch defects early, grade each roll with the four point system,
and turn every defect into data on machines, shifts and yarn suppliers.

## What is inside

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard with 5 tabs (inspect, roll report, analytics, cost, how it works) |
| `core.py` | Detection, four point grading, demo data, cost model in BDT |
| `train_fabric_yolo.ipynb` | Kaggle or Colab notebook that trains a YOLOv8 defect detector |
| `samples/` | Synthetic denim images so the app works offline |
| `models/` | Put your trained `best.pt` here |

## Step 1: Train the model (Kaggle, free GPU, about 30 minutes)

1. Go to kaggle.com, click **Create > New Notebook > File > Import Notebook**, upload `train_fabric_yolo.ipynb`.
2. In the right panel set **Accelerator: GPU T4** and turn **Internet on** (needs a phone verified account).
3. Run all cells. Option A downloads the Figshare Fabric Defects dataset (720 images; Oil, Hole, Cutting, Crack).
   If the download link fails, download `FD_Dataset.7z` by hand from
   https://figshare.com/articles/dataset/FD_Dataset_7z/25546465, upload it as a Kaggle dataset and use Option C.
4. Note the **mAP50** and per class numbers from step 5. Put them on a slide.
5. Download `best.pt` and `demo_images.zip` from the Output panel.
6. Copy `best.pt` into `models/` in this folder.

## Step 2: Run the app on your laptop

```bash
pip install -r requirements.txt
streamlit run app.py
```

Without `models/best.pt` the app uses a classical computer vision detector (texture anomaly check)
so the demo never breaks. With the model, the sidebar lets you switch between YOLO and classical.

## Demo script (5 minutes)

1. **Inspect image:** upload 2 or 3 images from `demo_images.zip`. Show boxes, labels, and penalty points.
   Say: "Each defect is measured in inches using camera calibration, then scored with the four point system."
2. **Roll report:** click *Run roll inspection*. Show the grade badge, points per 100 square yards,
   and the defect map. Say: "This map goes to the cutting section so markers avoid defects."
3. **Factory analytics:** point at Loom 07 night shift and Supplier B. Say: "This is simulated, but in a pilot
   this fills from every roll, and it tells maintenance and purchasing where to act."
4. **Cost impact:** ask them for their real numbers and type them in live. This turns the talk into their business.
5. **How it works:** pipeline and pilot plan.

## Be honest about these points

1. The public dataset is not Hameem fabric. A real pilot needs 2 to 4 weeks of image collection on their machines.
2. Roll frames in the demo are synthetic; analytics data is simulated. The cost model uses editable assumptions.
3. For new fabrics with few defect samples, the plan is an anomaly model (PatchCore via the Anomalib library)
   that learns only normal fabric.

## Dataset credit

Neamah, O. N. (2024). Fabric Defects Object Detection Dataset. Figshare. https://figshare.com/articles/dataset/FD_Dataset_7z/25546465
