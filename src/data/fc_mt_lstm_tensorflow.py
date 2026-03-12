"""
FC-MT-LSTM: Fairness-Constrained Multi-Task LSTM Model
TensorFlow/Keras Implementation
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import numpy as np


class SpatialCNN(layers.Layer):
    """
    Convolutional network for spatial feature extraction
    """
    def __init__(self, spatial_dim=64, **kwargs):
        super(SpatialCNN, self).__init__(**kwargs)
        self.spatial_dim = spatial_dim
        
        # Expand dims for 1D convolution
        self.expand = layers.Reshape((-1, 1))
        
        # Convolutional layers
        self.conv1 = layers.Conv1D(filters=32, kernel_size=3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        self.conv2 = layers.Conv1D(filters=64, kernel_size=3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        
        # Global pooling
        self.pool = layers.GlobalMaxPooling1D()
        
        # Dense projection
        self.dense = layers.Dense(spatial_dim, activation='relu')
    
    def call(self, inputs, training=False):
        x = self.expand(inputs)
        
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        
        x = self.pool(x)
        x = self.dense(x)
        
        return x


class TemporalLSTMWithAttention(layers.Layer):
    """
    LSTM with attention mechanism for temporal sequence encoding
    """
    def __init__(self, hidden_dim=128, **kwargs):
        super(TemporalLSTMWithAttention, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        
        # Bidirectional LSTM
        self.lstm = layers.Bidirectional(
            layers.LSTM(hidden_dim, return_sequences=True)
        )
        
        # Attention mechanism
        self.attention_dense = layers.Dense(1)
        self.attention_activation = layers.Activation('tanh')
        
    def call(self, inputs, training=False):
        # LSTM encoding
        lstm_out = self.lstm(inputs, training=training)  # (batch, time, 2*hidden_dim)
        
        # Attention scores
        attention_scores = self.attention_dense(lstm_out)  # (batch, time, 1)
        attention_scores = self.attention_activation(attention_scores)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)  # (batch, time, 1)
        
        # Weighted context vector
        context = tf.reduce_sum(lstm_out * attention_weights, axis=1)  # (batch, 2*hidden_dim)
        
        return context, attention_weights


class SharedEncoder(layers.Layer):
    """
    Fuses spatial and temporal representations
    """
    def __init__(self, encoding_dim=256, dropout_rate=0.3, **kwargs):
        super(SharedEncoder, self).__init__(**kwargs)
        
        self.dense1 = layers.Dense(encoding_dim)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.dropout1 = layers.Dropout(dropout_rate)
        
        self.dense2 = layers.Dense(encoding_dim)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, spatial_features, temporal_features, training=False):
        # Concatenate spatial and temporal
        x = tf.concat([spatial_features, temporal_features], axis=-1)
        
        # Dense layers
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.dropout2(x, training=training)
        
        return x


class GroupDecoder(layers.Layer):
    """
    Group-specific decoder for one protected group
    """
    def __init__(self, name_suffix, dropout_rate=0.2, **kwargs):
        super(GroupDecoder, self).__init__(name=f'decoder_{name_suffix}', **kwargs)
        
        self.dense1 = layers.Dense(128, activation='relu', name=f'{name_suffix}_dense1')
        self.bn1 = layers.BatchNormalization(name=f'{name_suffix}_bn1')
        self.dropout1 = layers.Dropout(dropout_rate, name=f'{name_suffix}_dropout1')
        
        self.dense2 = layers.Dense(64, activation='relu', name=f'{name_suffix}_dense2')
        self.bn2 = layers.BatchNormalization(name=f'{name_suffix}_bn2')
        self.dropout2 = layers.Dropout(dropout_rate, name=f'{name_suffix}_dropout2')
        
        self.output_layer = layers.Dense(1, activation='linear', name=f'{name_suffix}_output')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        output = self.output_layer(x)
        
        return output


class FC_MT_LSTM(Model):
    """
    Fairness-Constrained Multi-Task LSTM
    
    Architecture:
    - Spatial CNN for district features
    - Temporal LSTM with attention for time series
    - Shared encoder for feature fusion
    - 4 group-specific decoders (SC, ST, Women, Children)
    - Fairness-constrained loss function
    """
    def __init__(self, 
                 spatial_dim=64,
                 hidden_dim=128,
                 encoding_dim=256,
                 dropout_rate=0.3,
                 **kwargs):
        super(FC_MT_LSTM, self).__init__(**kwargs)
        
        # Components
        self.spatial_cnn = SpatialCNN(spatial_dim=spatial_dim)
        self.temporal_lstm = TemporalLSTMWithAttention(hidden_dim=hidden_dim)
        self.shared_encoder = SharedEncoder(encoding_dim=encoding_dim, dropout_rate=dropout_rate)
        
        # Group-specific decoders
        self.decoder_sc = GroupDecoder('sc')
        self.decoder_st = GroupDecoder('st')
        self.decoder_women = GroupDecoder('women')
        self.decoder_children = GroupDecoder('children')
        
        # Group mapping
        self.group_to_decoder = {
            'SC': self.decoder_sc,
            'ST': self.decoder_st,
            'Women': self.decoder_women,
            'Children': self.decoder_children
        }
        
        # Reverse mapping for numeric indices
        self.idx_to_group = {
            0: 'SC',
            1: 'ST',
            2: 'Women', 
            3: 'Children'
        }
    
    def call(self, inputs, training=False):
        """
        Forward pass
        
        Args:
            inputs: dict with keys:
                - 'spatial': (batch, spatial_features)
                - 'temporal': (batch, time_steps, temporal_features)
                - 'group': (batch,) group labels (can be strings or indices)
        
        Returns:
            predictions: (batch, 1)
            attention_weights: (batch, time_steps, 1)
        """
        spatial_input = inputs['spatial']
        temporal_input = inputs['temporal']
        group_labels = inputs['group']
        
        # Extract features
        spatial_features = self.spatial_cnn(spatial_input, training=training)
        temporal_features, attention_weights = self.temporal_lstm(temporal_input, training=training)
        
        # Fuse features
        shared_encoding = self.shared_encoder(spatial_features, temporal_features, training=training)
        
        # Handle different group label formats
        if group_labels.dtype == tf.string:
            # Convert string labels to predictions
            predictions = self._process_string_labels(shared_encoding, group_labels, training)
        else:
            # Convert integer labels to string and then predictions
            predictions = self._process_integer_labels(shared_encoding, group_labels, training)
        
        return predictions, attention_weights
    
    def _process_string_labels(self, shared_encoding, group_labels, training):
        """Process string group labels"""
        predictions = []
        for i in range(tf.shape(group_labels)[0]):
            group = group_labels[i]
            if isinstance(group, bytes):
                group_str = group.decode('utf-8')
            else:
                group_str = group.numpy().decode('utf-8') if hasattr(group, 'numpy') else str(group)
            
            decoder = self.group_to_decoder[group_str]
            pred = decoder(shared_encoding[i:i+1], training=training)
            predictions.append(pred)
        
        return tf.concat(predictions, axis=0)
    
    def _process_integer_labels(self, shared_encoding, group_labels, training):
        """Process integer group labels (0=SC, 1=ST, 2=Women, 3=Children)"""
        predictions = tf.TensorArray(dtype=tf.float32, size=tf.shape(group_labels)[0])
        
        for i in range(tf.shape(group_labels)[0]):
            group_idx = group_labels[i]
            group_str = self.idx_to_group[int(group_idx)]
            decoder = self.group_to_decoder[group_str]
            pred = decoder(shared_encoding[i:i+1], training=training)
            predictions = predictions.write(i, tf.squeeze(pred))
        
        return tf.expand_dims(predictions.stack(), axis=1)
    
    def predict_for_group(self, spatial_input, temporal_input, group, training=False):
        """
        Predict for a specific group
        
        Args:
            spatial_input: (batch, spatial_features)
            temporal_input: (batch, time_steps, temporal_features)
            group: str, one of ['SC', 'ST', 'Women', 'Children']
        
        Returns:
            predictions: (batch, 1)
        """
        # Extract features
        spatial_features = self.spatial_cnn(spatial_input, training=training)
        temporal_features, _ = self.temporal_lstm(temporal_input, training=training)
        
        # Fuse features
        shared_encoding = self.shared_encoder(spatial_features, temporal_features, training=training)
        
        # Decode
        decoder = self.group_to_decoder[group]
        predictions = decoder(shared_encoding, training=training)
        
        return predictions


def fairness_constrained_loss(y_true, y_pred, groups, lambda_fairness=1.0):
    """
    Custom fairness-constrained loss function
    
    Args:
        y_true: (batch, 1) actual values
        y_pred: (batch, 1) predictions  
        groups: (batch,) group labels
        lambda_fairness: weight for fairness penalty
    """
    # Prediction loss (MSE)
    pred_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Fairness penalty
    group_maes = {}
    for group_id, group_name in {0: 'SC', 1: 'ST', 2: 'Women', 3: 'Children'}.items():
        # Filter predictions for this group
        mask = tf.equal(groups, group_id)
        group_true = tf.boolean_mask(y_true, mask)
        group_pred = tf.boolean_mask(y_pred, mask)
        
        if tf.size(group_true) > 0:
            group_mae = tf.reduce_mean(tf.abs(group_true - group_pred))
            group_maes[group_name] = group_mae
    
    # Pairwise MAE differences
    fairness_penalty = 0.0
    group_list = list(group_maes.keys())
    for i in range(len(group_list)):
        for j in range(i+1, len(group_list)):
            fairness_penalty += tf.abs(group_maes[group_list[i]] - group_maes[group_list[j]])
    
    # Total loss
    total_loss = pred_loss + lambda_fairness * fairness_penalty
    
    return total_loss


class FairnessConstrainedTrainer:
    """
    Custom trainer with fairness-constrained loss
    """
    def __init__(self, model, lambda_fairness=1.0):
        self.model = model
        self.lambda_fairness = lambda_fairness
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Metrics
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.train_mae_metric = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
        
        self.val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
        self.val_mae_metric = tf.keras.metrics.MeanAbsoluteError(name='val_mae')
    
    def compute_loss(self, y_true, y_pred, groups):
        """
        Compute fairness-constrained loss
        
        Args:
            y_true: (batch, 1) actual values
            y_pred: (batch, 1) predictions
            groups: (batch,) group labels
        
        Returns:
            total_loss: scalar
            pred_loss: scalar
            fairness_loss: scalar
        """
        # Prediction loss (MSE)
        pred_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Fairness penalty
        group_maes = {}
        for group_id, group_name in {0: 'SC', 1: 'ST', 2: 'Women', 3: 'Children'}.items():
            # Filter predictions for this group
            mask = tf.equal(groups, group_id)
            group_true = tf.boolean_mask(y_true, mask)
            group_pred = tf.boolean_mask(y_pred, mask)
            
            if tf.size(group_true) > 0:
                group_mae = tf.reduce_mean(tf.abs(group_true - group_pred))
                group_maes[group_name] = group_mae
        
        # Pairwise MAE differences
        fairness_penalty = 0.0
        group_list = list(group_maes.keys())
        for i in range(len(group_list)):
            for j in range(i+1, len(group_list)):
                fairness_penalty += tf.abs(group_maes[group_list[i]] - group_maes[group_list[j]])
        
        # Total loss
        total_loss = pred_loss + self.lambda_fairness * fairness_penalty
        
        return total_loss, pred_loss, fairness_penalty
    
    @tf.function
    def train_step(self, spatial_batch, temporal_batch, y_batch, groups_batch):
        """
        Single training step
        """
        with tf.GradientTape() as tape:
            # Forward pass
            inputs = {
                'spatial': spatial_batch,
                'temporal': temporal_batch,
                'group': groups_batch
            }
            predictions, _ = self.model(inputs, training=True)
            
            # Compute loss
            total_loss, pred_loss, fairness_loss = self.compute_loss(
                y_batch, predictions, groups_batch
            )
        
        # Backward pass
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss_metric.update_state(total_loss)
        self.train_mae_metric.update_state(y_batch, predictions)
        
        return total_loss, pred_loss, fairness_loss
    
    @tf.function
    def val_step(self, spatial_batch, temporal_batch, y_batch, groups_batch):
        """
        Single validation step
        """
        inputs = {
            'spatial': spatial_batch,
            'temporal': temporal_batch,
            'group': groups_batch
        }
        predictions, _ = self.model(inputs, training=False)
        
        total_loss, pred_loss, fairness_loss = self.compute_loss(
            y_batch, predictions, groups_batch
        )
        
        self.val_loss_metric.update_state(total_loss)
        self.val_mae_metric.update_state(y_batch, predictions)
        
        return total_loss, pred_loss, fairness_loss
    
    def fit(self, train_dataset, val_dataset, epochs=100, patience=10):
        """
        Train the model
        
        Args:
            train_dataset: tf.data.Dataset
            val_dataset: tf.data.Dataset
            epochs: int
            patience: int for early stopping
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Reset metrics
            self.train_loss_metric.reset_states()
            self.train_mae_metric.reset_states()
            self.val_loss_metric.reset_states()
            self.val_mae_metric.reset_states()
            
            # Training loop
            for batch_idx, (spatial, temporal, y, groups) in enumerate(train_dataset):
                total_loss, pred_loss, fairness_loss = self.train_step(
                    spatial, temporal, y, groups
                )
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: Loss={total_loss:.4f}, "
                          f"Pred={pred_loss:.4f}, Fairness={fairness_loss:.4f}")
            
            # Validation loop
            for spatial, temporal, y, groups in val_dataset:
                self.val_step(spatial, temporal, y, groups)
            
            # Print epoch results
            train_loss = self.train_loss_metric.result()
            train_mae = self.train_mae_metric.result()
            val_loss = self.val_loss_metric.result()
            val_mae = self.val_mae_metric.result()
            
            print(f"\n  Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.model.save_weights('fc_mt_lstm_best.h5')
                print("  ✓ Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best weights
        self.model.load_weights('fc_mt_lstm_best.h5')
        print("\n✓ Training complete! Best model loaded.")


def evaluate_fc_mt_lstm(model, X_spatial_test, X_temporal_test, y_test, groups_test, 
                        target_scaler):
    """
    Comprehensive evaluation including fairness metrics
    
    Returns:
        results: dict with overall and per-group metrics
    """
    # Make predictions
    predictions = []
    attention_weights_all = []
    
    for i in range(len(X_spatial_test)):
        inputs = {
            'spatial': X_spatial_test[i:i+1],
            'temporal': X_temporal_test[i:i+1],
            'group': [groups_test[i]]
        }
        pred, attn = model(inputs, training=False)
        predictions.append(pred.numpy()[0, 0])
        attention_weights_all.append(attn.numpy()[0])
    
    predictions = np.array(predictions).reshape(-1, 1)
    
    # Inverse transform to original scale
    y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_orig = target_scaler.inverse_transform(predictions)
    
    # Overall metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    overall_mae = mean_absolute_error(y_test_orig, predictions_orig)
    overall_rmse = np.sqrt(mean_squared_error(y_test_orig, predictions_orig))
    overall_r2 = r2_score(y_test_orig, predictions_orig)
    
    print("="*80)
    print("OVERALL METRICS")
    print("="*80)
    print(f"MAE: {overall_mae:.4f}")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"R²: {overall_r2:.4f}")
    
    # Per-group metrics
    print("\n" + "="*80)
    print("PER-GROUP METRICS")
    print("="*80)
    
    group_metrics = {}
    mae_values = []
    
    for group_id, group_name in {0: 'SC', 1: 'ST', 2: 'Women', 3: 'Children'}.items():
        mask = groups_test == group_id
        group_true = y_test_orig[mask]
        group_pred = predictions_orig[mask]
        
        if len(group_true) > 0:
            group_mae = mean_absolute_error(group_true, group_pred)
            group_rmse = np.sqrt(mean_squared_error(group_true, group_pred))
            group_r2 = r2_score(group_true, group_pred)
            
            group_metrics[group_name] = {
                'mae': float(group_mae),
                'rmse': float(group_rmse),
                'r2': float(group_r2),
                'count': int(len(group_true))
            }
            
            mae_values.append(group_mae)
            
            print(f"\n{group_name}:")
            print(f"  MAE: {group_mae:.4f}")
            print(f"  RMSE: {group_rmse:.4f}")
            print(f"  R²: {group_r2:.4f}")
            print(f"  Samples: {len(group_true)}")
    
    # Fairness metrics
    fairness_gap = max(mae_values) - min(mae_values)
    fairness_ratio = max(mae_values) / min(mae_values) if min(mae_values) > 0 else float('inf')
    
    print("\n" + "="*80)
    print("FAIRNESS METRICS")
    print("="*80)
    print(f"Fairness Gap: {fairness_gap:.4f}")
    print(f"Fairness Ratio: {fairness_ratio:.2f}x")
    
    if fairness_gap < 0.5:
        print("✅ Excellent Fairness!")
    elif fairness_gap < 1.0:
        print("⚠️  Good Fairness")
    else:
        print("❌ Poor Fairness")
    
    results = {
        'overall': {
            'mae': overall_mae,
            'rmse': overall_rmse,
            'r2': overall_r2
        },
        'by_group': group_metrics,
        'fairness': {
            'gap': fairness_gap,
            'ratio': fairness_ratio
        },
        'predictions': predictions_orig,
        'attention_weights': attention_weights_all
    }
    
    return results


if __name__ == "__main__":
    # Test the model architecture
    print("Testing FC-MT-LSTM TensorFlow Model Architecture...")
    
    # Create model
    model = FC_MT_LSTM(
        spatial_dim=64,
        hidden_dim=128,
        encoding_dim=256,
        dropout_rate=0.3
    )
    
    # Sample inputs
    batch_size = 32
    spatial_features = 10
    temporal_seq_len = 5
    temporal_features = 20
    
    spatial_input = tf.random.normal((batch_size, spatial_features))
    temporal_input = tf.random.normal((batch_size, temporal_seq_len, temporal_features))
    group_labels = tf.constant([0, 1, 2, 3] * (batch_size // 4))  # Alternate groups
    
    inputs = {
        'spatial': spatial_input,
        'temporal': temporal_input,
        'group': group_labels
    }
    
    predictions, attention_weights = model(inputs, training=False)
    
    print(f"Spatial input shape: {spatial_input.shape}")
    print(f"Temporal input shape: {temporal_input.shape}")
    print(f"Group labels shape: {group_labels.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print("✅ TensorFlow model architecture test passed!")
