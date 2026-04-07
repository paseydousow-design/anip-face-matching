# ANIP - Tache 1 : Face Matching / Reconnaissance Faciale

Systeme de reconnaissance faciale developpe dans le cadre du challenge ANIP.  
Objectif : identifier si deux photos representent la meme personne, parmi **~2 millions de paires** d'images.

---

## Iterations (V2 → V9)

| Version | Approche | Changement cle |
|---------|----------|----------------|
| V2 | Reseau siamois + MobileNetV2 (TensorFlow) | Architecture de base |
| V3 | + Augmentation de donnees | Flip, brightness, contrast |
| V4 | + Fine-tuning des dernieres couches | Degelage partiel du backbone |
| V6 | + Triplet Loss | Meilleure separation des embeddings |
| V7 | FaceNet VGGFace2 (PyTorch) | Passage aux modeles specialises visages |
| V8 | + ArcFace + TTA 5 vues | Ensemble de modeles, augmentation au test |
| **V9** | **+ Correction bayesienne du prior** | **Resolution du prior mismatch** |

---

## Le probleme cle resolu en V9

Le jeu de calibration etait **50% positif / 50% negatif**.  
Le test reel etait **~0.05% positif** (1 000 vraies paires sur 2 000 000).

Appliquer naïvement le seuil calibre → **170 255 faux matchs**.  
Avec la correction du prior bayesien → **3 366 matchs coherents**.

---

## Stack technique (V9 finale)

- **PyTorch** — FaceNet VGGFace2 (embeddings 512-dim)
- **insightface** — ArcFace R100 (optionnel)
- **TTA 5 vues** : original, flip horizontal, rotation +10°/-10°, luminosite +20
- **Similarite cosinus** sur embeddings L2-normalises
- **Seuil optimal** : indice de Youden + correction bayesienne du prior

---

## Structure du projet

```
anip-face-matching/
├── notebooks/
│   ├── tache1_improved_v2.ipynb   # Siamois MobileNetV2
│   ├── tache1_improved_v3.ipynb
│   ├── tache1_improved_v4.ipynb
│   ├── tache1_improved_v6.ipynb
│   ├── tache1_improved_v7.ipynb
│   ├── tache1_improved_v8.ipynb
│   └── tache1_v9_final.ipynb      # Version finale
├── outputs/
│   ├── calibration_v7/v8/v9.png   # Courbes ROC & calibration
│   ├── matches_v7/v8/v9.png       # Paires matchees
│   ├── diagnostic_v8.png
│   └── tache1_submission_v9.csv   # Fichier de soumission final
├── src/
│   └── tache1_face_matching.py
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Pour ArcFace (optionnel) :
```bash
pip install insightface onnxruntime
```

---

## Resultats

- **AUC-ROC** : 0.7575
- **Paires matchees** : 3 366 / 1 999 000 (~0.17%)
- **Seuil final** : 0.554 (prior-corrige)
