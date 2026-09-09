# Chapter 1: Data Ingestion & XML Parsing — The 15-Line Truth

In the project codebase, [`data/scripts/convert_pets2009.py`](file:///D:/yasmin%20programs/PROGLINT/data/scripts/convert_pets2009.py) is over 600 lines long. 

When you look at 600 lines of code, it feels overwhelming. But just like training was only 5 lines, **the actual data conversion logic is literally 15 lines of simple Python**.

---

## 1. The Pure 15-Line Conversion Code

Here is the exact code that converts raw PETS-2009 XML annotations into YOLO/RT-DETR training format:

```python
import xml.etree.ElementTree as ET

def convert_xml_to_yolo(xml_file, output_txt, img_w=768, img_h=576):
    """The 15-line core: reads XML, divides by width/height, writes text file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    with open(output_txt, "w") as out:
        for obj in root.findall(".//object"):
            box = obj.find("box")
            if box is not None:
                # 1. Read pixel values from XML
                xc, yc = float(box.get("xc")), float(box.get("yc"))
                w,  h  = float(box.get("w")),  float(box.get("h"))
                
                # 2. Normalize to [0.0, 1.0] by dividing by image dimensions
                n_xc, n_yc = xc / img_w, yc / img_h
                n_w,  n_h  = w  / img_w, h  / img_h
                
                # 3. Write YOLO line: "0 <xc> <yc> <w> <h>" (0 = person)
                out.write(f"0 {n_xc:.6f} {n_yc:.6f} {n_w:.6f} {n_h:.6f}\n")
```

**That is the whole conversion logic.**

### What Happens In Those 15 Lines?
1. `ET.parse(xml_file)`: Python's built-in XML reader opens the file.
2. `root.findall(".//object")`: Finds every pedestrian tag `<object>`.
3. `xc / img_w`: Divides pixel center $X$ by image width ($768$) to get a number between $0.0$ and $1.0$.
4. `yc / img_h`: Divides pixel center $Y$ by image height ($576$) to get a number between $0.0$ and $1.0$.
5. `out.write(f"0 ...")`: Writes single-class pedestrian labels that RT-DETR can read.

---

## 2. Why Did `convert_pets2009.py` Have 600 Lines?

The other 585 lines were purely automated convenience features:
1. **Automated downloader:** Code that downloads `.tar.gz` files from the internet.
2. **Decompressor:** Code that unpacks `.tar` archives and creates directories automatically.
3. **Sequence Splitter:** Code that moves `S1L1` into `images/train/` and `S1L2` into `images/val/`.
4. **Drawing verification:** Code that uses OpenCV to draw boxes on validation images to save `sample_verification.jpg`.

**What you need to remember:**  
All "dataset preparation" really means is: **Read the raw XML coordinates, divide by image width/height to normalize, and save as `.txt` files.**

---

## 3. The Only Other File Needed: `pets2009.yaml` (6 Lines)

After generating the `.txt` label files, you create a text file called [`pets2009.yaml`](file:///D:/yasmin%20programs/PROGLINT/data/processed/pets2009.yaml):

```yaml
path: data/processed       # Where the folder is
train: images/train        # Where training images are
val: images/val            # Where validation images are
names:
  0: person                # What class 0 means
```

That's it! Your dataset is ready for training.
