"""
ANIP - Tâche 1 : Reconnaissance Faciale Robuste
Face Matching/Verification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.layers import Dense, Lambda, GlobalAveragePooling2D, Dropout
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine, euclidean

print("🎯 ANIP - Tâche 1: Reconnaissance Faciale\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = Path('anip-reconnaissance-faciale-estimation-ages-ocr/dataset_tache_1/dataset_tache_1')
TRAIN_PATH = DATA_PATH / 'train'
TEST_PATH = DATA_PATH / 'test'

IMG_SIZE = (160, 160)  # Taille standard pour face recognition
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================================
# 1. CHARGEMENT ET ANALYSE DES DONNÉES
# ============================================================================

def parse_train_filename(filepath):
    """
    Parse le nom de fichier du train set
    Format: XXXX_Y.jpg où XXXX=person_id, Y=photo_num (0 ou 1)
    """
    filename = filepath.stem
    parts = filename.split('_')
    person_id = int(parts[0])
    photo_num = int(parts[1])
    return person_id, photo_num

def load_train_data():
    """Charge les données d'entraînement"""
    train_images = list(TRAIN_PATH.glob('*.jpg')) + list(TRAIN_PATH.glob('*.JPG'))
    
    data = []
    for img_path in train_images:
        person_id, photo_num = parse_train_filename(img_path)
        data.append({
            'filepath': str(img_path),
            'person_id': person_id,
            'photo_num': photo_num
        })
    
    df = pd.DataFrame(data)
    return df

print("📂 Chargement des données...")
df_train = load_train_data()

print(f"✅ Images d'entraînement: {len(df_train)}")
print(f"   Personnes uniques: {df_train['person_id'].nunique()}")
print(f"   Photos par personne: {df_train.groupby('person_id').size().mode()[0]}")

# ============================================================================
# 2. CRÉATION DE PAIRES POSITIVES ET NÉGATIVES
# ============================================================================

def create_pairs(df, n_positive=1000, n_negative=1000):
    """
    Crée des paires d'images pour l'entraînement
    - Paires positives : même personne
    - Paires négatives : personnes différentes
    """
    pairs = []
    labels = []
    
    # Paires positives (même personne)
    print("Création des paires positives...")
    person_ids = df['person_id'].unique()
    
    for _ in tqdm(range(n_positive)):
        person_id = np.random.choice(person_ids)
        person_imgs = df[df['person_id'] == person_id]['filepath'].values
        
        if len(person_imgs) >= 2:
            img1, img2 = np.random.choice(person_imgs, 2, replace=False)
            pairs.append([img1, img2])
            labels.append(1)  # Même personne
    
    # Paires négatives (personnes différentes)
    print("Création des paires négatives...")
    for _ in tqdm(range(n_negative)):
        person_id1, person_id2 = np.random.choice(person_ids, 2, replace=False)
        
        img1 = df[df['person_id'] == person_id1]['filepath'].values[0]
        img2 = df[df['person_id'] == person_id2]['filepath'].values[0]
        
        pairs.append([img1, img2])
        labels.append(0)  # Personnes différentes
    
    return np.array(pairs), np.array(labels)

# Créer les paires
pairs, labels = create_pairs(df_train, n_positive=2000, n_negative=2000)
print(f"\n✅ Paires créées: {len(pairs)}")
print(f"   Positives: {labels.sum()}")
print(f"   Négatives: {len(labels) - labels.sum()}")

# Split train/validation
pairs_train, pairs_val, labels_train, labels_val = train_test_split(
    pairs, labels, test_size=0.2, random_state=SEED
)

# ============================================================================
# 3. GÉNÉRATEUR DE DONNÉES
# ============================================================================

def load_and_preprocess_image(image_path, img_size=IMG_SIZE):
    """Charge et prétraite une image"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.astype('float32') / 255.0
    return img

def pair_generator(pairs, labels, batch_size=32, shuffle=True):
    """Générateur de paires d'images"""
    n_samples = len(pairs)
    indices = np.arange(n_samples)
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx+batch_size]
            
            batch_pairs = pairs[batch_indices]
            batch_labels = labels[batch_indices]
            
            # Charger les images
            batch_img1 = np.array([load_and_preprocess_image(p[0]) for p in batch_pairs])
            batch_img2 = np.array([load_and_preprocess_image(p[1]) for p in batch_pairs])
            
            yield [batch_img1, batch_img2], batch_labels

# ============================================================================
# 4. MODÈLE SIAMOIS (SIAMESE NETWORK)
# ============================================================================

def create_base_network(input_shape):
    """Réseau de base pour extraire les features"""
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    embeddings = Dense(128, activation='linear')(x)  # Embedding vector
    
    model = Model(inputs=inputs, outputs=embeddings, name='base_network')
    return model

def create_siamese_network(input_shape):
    """Crée un réseau siamois pour la vérification de visages"""
    
    # Réseau de base partagé
    base_network = create_base_network(input_shape)
    
    # Entrées
    input_a = Input(shape=input_shape, name='input_a')
    input_b = Input(shape=input_shape, name='input_b')
    
    # Passer les deux images dans le même réseau
    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)
    
    # Calculer la distance L1
    l1_distance = Lambda(lambda x: tf.abs(x[0] - x[1]))([embedding_a, embedding_b])
    
    # Prédiction finale
    prediction = Dense(1, activation='sigmoid')(l1_distance)
    
    # Créer le modèle
    siamese_model = Model(inputs=[input_a, input_b], outputs=prediction)
    
    return siamese_model, base_network

# Créer le modèle
print("\n🧠 Création du modèle siamois...")
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
siamese_model, base_network = create_siamese_network(input_shape)

siamese_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✅ Modèle créé")
siamese_model.summary()

# ============================================================================
# 5. ENTRAÎNEMENT
# ============================================================================

print("\n🎯 Entraînement du modèle...")

steps_per_epoch = len(pairs_train) // BATCH_SIZE
validation_steps = len(pairs_val) // BATCH_SIZE

train_gen = pair_generator(pairs_train, labels_train, BATCH_SIZE)
val_gen = pair_generator(pairs_val, labels_val, BATCH_SIZE, shuffle=False)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_siamese_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

history = siamese_model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Entraînement terminé!")

# ============================================================================
# 6. EXTRACTION D'EMBEDDINGS
# ============================================================================

def get_embeddings_for_all_images(df, base_network, batch_size=32):
    """Calcule les embeddings pour toutes les images"""
    embeddings = []
    filepaths = df['filepath'].values
    
    print("\n📊 Extraction des embeddings...")
    for i in tqdm(range(0, len(filepaths), batch_size)):
        batch_paths = filepaths[i:i+batch_size]
        batch_images = np.array([load_and_preprocess_image(p) for p in batch_paths])
        
        batch_embeddings = base_network.predict(batch_images, verbose=0)
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

# Calculer les embeddings pour le train set
train_embeddings = get_embeddings_for_all_images(df_train, base_network)

# Ajouter les embeddings au DataFrame
df_train['embedding'] = list(train_embeddings)

print("✅ Embeddings calculés")

# ============================================================================
# 7. PRÉDICTION SUR LE TEST SET
# ============================================================================

def find_matches_in_test(test_path, base_network, threshold=0.5):
    """
    Trouve les paires correspondantes dans le test set
    """
    test_images = list(test_path.glob('*.jpg')) + list(test_path.glob('*.JPG'))
    print(f"\n🔍 Test images: {len(test_images)}")
    
    # Calculer les embeddings pour le test
    test_data = []
    test_embeddings = []
    
    print("Extraction des embeddings du test set...")
    for img_path in tqdm(test_images):
        img = load_and_preprocess_image(str(img_path))
        embedding = base_network.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        
        test_data.append({
            'filepath': str(img_path),
            'filename': img_path.name,
            'embedding': embedding
        })
        test_embeddings.append(embedding)
    
    df_test = pd.DataFrame(test_data)
    
    # Trouver les paires
    print("\nRecherche de paires...")
    matches = []
    n_test = len(df_test)
    
    for i in tqdm(range(n_test)):
        for j in range(i+1, n_test):
            emb1 = df_test.iloc[i]['embedding']
            emb2 = df_test.iloc[j]['embedding']
            
            # Calculer la similarité cosine
            similarity = 1 - cosine(emb1, emb2)
            
            if similarity > threshold:
                matches.append({
                    'image1': df_test.iloc[i]['filename'],
                    'image2': df_test.iloc[j]['filename'],
                    'similarity': similarity,
                    'is_match': 1
                })
    
    matches_df = pd.DataFrame(matches)
    return matches_df, df_test

# Prédire sur le test set
matches_df, df_test = find_matches_in_test(TEST_PATH, base_network, threshold=0.6)

print(f"\n✅ Paires trouvées: {len(matches_df)}")

# ============================================================================
# 8. CRÉATION DU FICHIER DE SOUMISSION
# ============================================================================

# Sauvegarder les résultats
matches_df.to_csv('tache1_submission.csv', index=False)
print(f"\n✅ Soumission sauvegardée: tache1_submission.csv")
print(f"\n📊 Aperçu:")
print(matches_df.head(10))

# ============================================================================
# 9. VISUALISATION DES RÉSULTATS
# ============================================================================

print("\n📊 Visualisation de quelques paires...")

# Sélectionner quelques paires
sample_matches = matches_df.sort_values('similarity', ascending=False).head(6)

fig, axes = plt.subplots(3, 4, figsize=(15, 12))
axes = axes.flatten()

for idx, (_, row) in enumerate(sample_matches.iterrows()):
    if idx >= 3:
        break
    
    img1_path = TEST_PATH / row['image1']
    img2_path = TEST_PATH / row['image2']
    
    img1 = cv2.imread(str(img1_path))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    img2 = cv2.imread(str(img2_path))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    axes[idx*2].imshow(img1)
    axes[idx*2].set_title(f"{row['image1']}")
    axes[idx*2].axis('off')
    
    axes[idx*2+1].imshow(img2)
    axes[idx*2+1].set_title(f"{row['image2']}\nSim: {row['similarity']:.3f}")
    axes[idx*2+1].axis('off')

plt.tight_layout()
plt.savefig('tache1_matches_visualization.png', dpi=150, bbox_inches='tight')
print("✅ Visualisation sauvegardée: tache1_matches_visualization.png")

print("\n" + "="*70)
print("✅ TÂCHE 1 TERMINÉE!")
print("="*70)
print(f"\nFichiers générés:")
print(f"  1. tache1_submission.csv")
print(f"  2. best_siamese_model.h5")
print(f"  3. tache1_matches_visualization.png")
print(f"\nProchaines étapes:")
print(f"  - Ajuster le threshold de similarité")
print(f"  - Essayer différents modèles (FaceNet, ArcFace)")
print(f"  - Ajouter la détection/alignement de visages")
print(f"  - Tester différentes métriques de distance")
