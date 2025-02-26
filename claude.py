import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2

# Define the class labels and corresponding RGB values
CLASS_LABELS = {
    'background': [0, 0, 0],       # Background
    'right_arm': [21, 21, 21],     # Right Arm
    'left_arm': [22, 22, 22],      # Left Arm
    'chest_middle': [5, 5, 5],     # Chest/Middle
    'collar_front': [24, 24, 24],  # Collar (Front)
    'body_back': [25, 25, 25]      # Body Back Parts
}

# Convert RGB labels to integers for one-hot encoding
LABEL_TO_INDEX = {
    (0, 0, 0): 0,      # Background
    (21, 21, 21): 1,   # Right Arm
    (22, 22, 22): 2,   # Left Arm
    (5, 5, 5): 3,      # Chest/Middle
    (24, 24, 24): 4,   # Collar (Front)
    (25, 25, 25): 5    # Body Back Parts
}

NUM_CLASSES = len(CLASS_LABELS)

# Define input image dimensions
IMG_HEIGHT = 768
IMG_WIDTH = 1024
IMG_CHANNELS = 3

def data_generator(input_img_paths, mask_img_paths, target_img_paths, batch_size=8):
    """Generator function to yield batches of data with input images, binary masks and segmentation targets"""
    num_samples = len(input_img_paths)
    while True:
        # Generate random batch of indices
        indices = np.random.randint(0, num_samples, batch_size)
        
        # Create empty arrays to hold batch of input, masks and output images
        batch_input_img_paths = [input_img_paths[i] for i in indices]
        batch_mask_img_paths = [mask_img_paths[i] for i in indices]
        batch_target_img_paths = [target_img_paths[i] for i in indices]
        
        batch_input_imgs = []
        batch_mask_imgs = []
        batch_target_imgs = []
        
        for i in range(batch_size):
            # Load input image and normalize
            input_img = cv2.imread(batch_input_img_paths[i])
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_img = input_img / 255.0
            
            # Load binary mask and normalize
            mask_img = cv2.imread(batch_mask_img_paths[i], cv2.IMREAD_GRAYSCALE)
            mask_img = mask_img / 255.0
            mask_img = np.expand_dims(mask_img, axis=-1)  # Add channel dimension
            
            # Load target segmentation image
            target_img = cv2.imread(batch_target_img_paths[i])
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            
            # Convert RGB labels to class indices
            seg_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
            for rgb, idx in LABEL_TO_INDEX.items():
                # Create binary mask for current rgb value
                binary_mask = np.all(target_img == rgb, axis=-1)
                # Assign class index to the mask
                seg_mask[binary_mask] = idx
            
            # Convert to one-hot encoding
            target_one_hot = to_categorical(seg_mask, num_classes=NUM_CLASSES)
            
            batch_input_imgs.append(input_img)
            batch_mask_imgs.append(mask_img)
            batch_target_imgs.append(target_one_hot)
        
        yield [np.array(batch_input_imgs), np.array(batch_mask_imgs)], np.array(batch_target_imgs)

def build_enhanced_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), n_classes=NUM_CLASSES):
    """
    Build enhanced U-Net model architecture that takes both the image and binary mask as input
    The binary mask helps focus the network on the garment region
    """
    # Input layers
    img_input = Input(input_shape)
    mask_input = Input((IMG_HEIGHT, IMG_WIDTH, 1))
    
    # Apply mask to input image to focus on garment
    masked_input = multiply([img_input, mask_input])
    
    # Contraction path (Encoder)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(masked_input)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = Dropout(0.3)(c5)
    
    # Expansion path (Decoder)
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    # Attention gate for focusing on garment boundaries
    attention_gate = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    c9 = multiply([c9, attention_gate])
    
    # Mask-aware output layer
    # We'll multiply by the mask again to ensure predictions are only within the garment area
    outputs_raw = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
    
    # Apply the binary mask to output to ensure we only predict labels within the garment area
    # This ensures background stays background
    mask_expanded = tf.repeat(mask_input, n_classes, axis=-1)
    background_mask = 1.0 - mask_expanded
    background_only = tf.concat([tf.ones_like(mask_input), tf.zeros_like(mask_expanded[..., 1:])], axis=-1)
    
    masked_output = outputs_raw * mask_expanded + background_only * background_mask
    
    # Ensure softmax sums to 1 for each pixel
    outputs = tf.keras.layers.Lambda(lambda x: x / tf.reduce_sum(x, axis=-1, keepdims=True))(masked_output)
    
    model = Model(inputs=[img_input, mask_input], outputs=outputs)
    
    # Define custom loss that ignores background pixels in accuracy calculation
    def weighted_categorical_crossentropy(y_true, y_pred):
        # Standard categorical crossentropy
        cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Create a mask of the non-background pixels (class 0 is background)
        mask = 1.0 - y_true[..., 0]
        
        # Weight the loss to focus on garment areas
        weighted_cce = cce * mask
        
        # Return the mean over all pixels
        return tf.reduce_mean(weighted_cce)
    
    # Custom accuracy metric that only considers garment pixels
    def garment_accuracy(y_true, y_pred):
        # Get predicted class indices
        y_pred_class = tf.argmax(y_pred, axis=-1)
        y_true_class = tf.argmax(y_true, axis=-1)
        
        # Create a mask for non-background pixels (where true class != 0)
        mask = tf.cast(y_true_class > 0, tf.float32)
        
        # Calculate accuracy only on garment pixels
        correct_pixels = tf.cast(tf.equal(y_pred_class, y_true_class), tf.float32) * mask
        total_garment_pixels = tf.reduce_sum(mask)
        
        # Avoid division by zero
        return tf.cond(
            tf.equal(total_garment_pixels, 0),
            lambda: tf.constant(1.0, dtype=tf.float32),
            lambda: tf.reduce_sum(correct_pixels) / total_garment_pixels
        )
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4), 
        loss=weighted_categorical_crossentropy,
        metrics=['accuracy', garment_accuracy]
    )
    
    return model

def train_model(input_dir, mask_dir, output_dir, model_save_path, batch_size=8, epochs=100):
    """Train the segmentation model using input images, binary masks, and target segmentations"""
    # Get file paths
    input_img_paths = sorted([
        os.path.join(input_dir, fname) 
        for fname in os.listdir(input_dir) 
        if fname.endswith('.jpg') or fname.endswith('.png')
    ])
    
    mask_img_paths = sorted([
        os.path.join(mask_dir, fname) 
        for fname in os.listdir(mask_dir) 
        if fname.endswith('.jpg') or fname.endswith('.png')
    ])
    
    target_img_paths = sorted([
        os.path.join(output_dir, fname) 
        for fname in os.listdir(output_dir) 
        if fname.endswith('.jpg') or fname.endswith('.png')
    ])
    
    print(f"Found {len(input_img_paths)} images for training")
    
    # Make sure all three directories have the same number of files
    assert len(input_img_paths) == len(mask_img_paths) == len(target_img_paths), \
        "Input, mask, and output directories must contain the same number of files"
    
    # Split data into training and validation sets
    train_input_paths, val_input_paths, train_mask_paths, val_mask_paths, train_target_paths, val_target_paths = train_test_split(
        input_img_paths, mask_img_paths, target_img_paths, test_size=0.2, random_state=42
    )
    
    # Create data generators
    train_gen = data_generator(train_input_paths, train_mask_paths, train_target_paths, batch_size)
    val_gen = data_generator(val_input_paths, val_mask_paths, val_target_paths, batch_size)
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_input_paths) // batch_size
    validation_steps = len(val_input_paths) // batch_size
    
    # Build model
    model = build_enhanced_unet()
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_garment_accuracy'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-7),
        EarlyStopping(patience=15, monitor='val_garment_accuracy', verbose=1, mode='max', restore_best_weights=True)
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    return model, history

def predict_and_visualize(model, image_path, mask_path=None):
    """Predict segmentation for a single image and visualize results"""
    # Load and preprocess input image
    input_img = cv2.imread(image_path)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img_norm = input_img / 255.0
    
    # If mask path is provided, use it; otherwise generate a simple mask of all 1s
    if mask_path:
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_img = mask_img / 255.0
    else:
        # If no mask, create one with 1s (assume the whole image could contain garment)
        mask_img = np.ones((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    
    mask_img = np.expand_dims(mask_img, axis=-1)  # Add channel dimension
    
    # Expand dimensions to match model input shape
    input_img_expanded = np.expand_dims(input_img_norm, axis=0)
    mask_img_expanded = np.expand_dims(mask_img, axis=0)
    
    # Predict
    prediction = model.predict([input_img_expanded, mask_img_expanded])
    
    # Get class index with highest probability for each pixel
    predicted_mask = np.argmax(prediction[0], axis=-1)
    
    # Convert class indices to RGB values
    segmentation_map = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    
    # Map indices back to RGB values
    for label_idx, rgb_value in enumerate(CLASS_LABELS.values()):
        mask = predicted_mask == label_idx
        segmentation_map[mask] = rgb_value
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(input_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Predicted Segmentation")
    plt.imshow(segmentation_map)
    plt.axis('off')
    
    # Create colored overlay
    overlay = input_img.copy()
    alpha = 0.4
    mask = predicted_mask > 0  # Non-background pixels
    overlay[mask] = overlay[mask] * (1 - alpha) + segmentation_map[mask] * alpha
    
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return segmentation_map, overlay

def visualize_dataset_samples(input_dir, mask_dir, output_dir, sample_size=3):
    """Visualize samples from the dataset to verify data loading"""
    # Get random sample of file paths
    input_img_paths = sorted([
        os.path.join(input_dir, fname) 
        for fname in os.listdir(input_dir) 
        if fname.endswith('.jpg') or fname.endswith('.png')
    ])
    
    mask_img_paths = sorted([
        os.path.join(mask_dir, fname) 
        for fname in os.listdir(mask_dir) 
        if fname.endswith('.jpg') or fname.endswith('.png')
    ])
    
    target_img_paths = sorted([
        os.path.join(output_dir, fname) 
        for fname in os.listdir(output_dir) 
        if fname.endswith('.jpg') or fname.endswith('.png')
    ])
    
    # Select random samples
    indices = np.random.choice(len(input_img_paths), sample_size, replace=False)
    
    for i, idx in enumerate(indices):
        # Load input image
        input_img = cv2.imread(input_img_paths[idx])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # Load binary mask
        mask_img = cv2.imread(mask_img_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Load target image (segmentation)
        target_img = cv2.imread(target_img_paths[idx])
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        
        # Apply mask to input for visualization
        masked_input = input_img.copy()
        masked_input[mask_img == 0] = 0
        
        # Display
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.title("Input Image")
        plt.imshow(input_img)
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.title("Binary Mask")
        plt.imshow(mask_img, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.title("Masked Input")
        plt.imshow(masked_input)
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.title("Target Segmentation")
        plt.imshow(target_img)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def load_model_and_predict(model_path, image_path, mask_path=None):
    """Load a trained model and make predictions on a new image"""
    # Load the model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Recompile the model with custom metrics (if needed)
    model.compile(optimizer=Adam(learning_rate=1e-4), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Make predictions
    return predict_and_visualize(model, image_path, mask_path)

# Data augmentation function to increase training dataset variety
def augment_dataset(input_dir, mask_dir, output_dir, augmented_input_dir, augmented_mask_dir, augmented_output_dir, samples=1000):
    """Augment the dataset with random transformations to improve model generalization"""
    # Create output directories if they don't exist
    os.makedirs(augmented_input_dir, exist_ok=True)
    os.makedirs(augmented_mask_dir, exist_ok=True)
    os.makedirs(augmented_output_dir, exist_ok=True)
    
    # Get file paths
    input_img_paths = sorted([
        os.path.join(input_dir, fname) 
        for fname in os.listdir(input_dir) 
        if fname.endswith('.jpg') or fname.endswith('.png')
    ])
    
    mask_img_paths = sorted([
        os.path.join(mask_dir, fname) 
        for fname in os.listdir(mask_dir) 
        if fname.endswith('.jpg') or fname.endswith('.png')
    ])
    
    target_img_paths = sorted([
        os.path.join(output_dir, fname) 
        for fname in os.listdir(output_dir) 
        if fname.endswith('.jpg') or fname.endswith('.png')
    ])
    
    # Define augmentation operations
    def apply_augmentation(input_img, mask_img, target_img):
        # Randomly select augmentation operations
        # 1. Horizontal flip
        if np.random.rand() > 0.5:
            input_img = cv2.flip(input_img, 1)
            mask_img = cv2.flip(mask_img, 1)
            target_img = cv2.flip(target_img, 1)
        
        # 2. Small rotation (+/- 10 degrees)
        angle = np.random.uniform(-10, 10)
        h, w = input_img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        input_img = cv2.warpAffine(input_img, M, (w, h))
        mask_img = cv2.warpAffine(mask_img, M, (w, h))
        target_img = cv2.warpAffine(target_img, M, (w, h))
        
        # 3. Small brightness/contrast adjustments (only for input image)
        alpha = np.random.uniform(0.8, 1.2)  # Contrast
        beta = np.random.uniform(-30, 30)    # Brightness
        input_img = cv2.convertScaleAbs(input_img, alpha=alpha, beta=beta)
        
        return input_img, mask_img, target_img
    
    # Generate augmented samples
    for i in range(samples):
        # Randomly select an image
        idx = np.random.randint(0, len(input_img_paths))
        
        # Read images
        input_img = cv2.imread(input_img_paths[idx])
        mask_img = cv2.imread(mask_img_paths[idx], cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(target_img_paths[idx])
        
        # Apply augmentation
        augmented_input, augmented_mask, augmented_target = apply_augmentation(input_img, mask_img, target_img)
        
        # Save augmented images
        output_filename = f"aug_{i:04d}.png"
        cv2.imwrite(os.path.join(augmented_input_dir, output_filename), augmented_input)
        cv2.imwrite(os.path.join(augmented_mask_dir, output_filename), augmented_mask)
        cv2.imwrite(os.path.join(augmented_output_dir, output_filename), augmented_target)
        
    print(f"Generated {samples} augmented samples")


# Example usage
if __name__ == "__main__":
    # Replace these with your actual paths
    input_dir = "path/to/input/images"
    mask_dir = "path/to/binary/masks"
    output_dir = "path/to/segmentation/masks"
    model_save_path = "garment_segmentation_model.h5"
    
    # Visualize some samples from dataset
    # visualize_dataset_samples(input_dir, mask_dir, output_dir)
    
    # Optional: Augment dataset to improve model performance
    # augment_dataset(input_dir, mask_dir, output_dir, 
    #                "augmented/input", "augmented/mask", "augmented/output", 
    #                samples=2000)
    
    # Train model
    model, history = train_model(input_dir, mask_dir, output_dir, model_save_path, batch_size=8, epochs=100)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['garment_accuracy'])
    plt.plot(history.history['val_garment_accuracy'])
    plt.title('Garment Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.show()
    
    # Test on a single image
    test_image_path = "path/to/test/image.jpg"
    test_mask_path = "path/to/test/mask.jpg"
    segmentation_map, overlay = predict_and_visualize(model, test_image_path, test_mask_path)