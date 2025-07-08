#!/usr/bin/env python3
"""
Yüz Maskesi Tespit - Basit Test Script
"""
print("🚀 Yüz Maskesi Tespit Projesi Başlatılıyor...")

try:
    import os
    import random
    import numpy as np
    import cv2
    import matplotlib
    matplotlib.use('Agg')  # GUI olmayan backend
    import matplotlib.pyplot as plt
    print("✅ Gerekli kütüphaneler yüklendi")
    
    # TensorFlow import
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} hazır")
    
    # Proje klasörü
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 Proje klasörü: {BASE_DIR}")
    
    # Model dosyası kontrolü
    MODEL_PATH = os.path.join(BASE_DIR, "models", "face_mask_detector.h5")
    if os.path.exists(MODEL_PATH):
        model_size = os.path.getsize(MODEL_PATH)
        print(f"✅ Model dosyası bulundu ({model_size:,} bytes)")
    else:
        print("❌ Model dosyası bulunamadı!")
        exit(1)
    
    # Veri klasörleri kontrolü
    DATA_DIR = os.path.join(BASE_DIR, "data")
    with_mask_dir = os.path.join(DATA_DIR, "with_mask")
    without_mask_dir = os.path.join(DATA_DIR, "without_mask")
    
    if os.path.exists(with_mask_dir):
        with_mask_count = len([f for f in os.listdir(with_mask_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"✅ Maskeli görseller: {with_mask_count} adet")
    else:
        print("❌ Maskeli görsel klasörü bulunamadı!")
        exit(1)
    
    if os.path.exists(without_mask_dir):
        without_mask_count = len([f for f in os.listdir(without_mask_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"✅ Maskesiz görseller: {without_mask_count} adet")
    else:
        print("❌ Maskesiz görsel klasörü bulunamadı!")
        exit(1)
    
    print("\n📊 Model yükleniyor...")
    
    # Model yükleme
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model başarıyla yüklendi!")
        
        # Model yapısı bilgisi
        input_shape = model.input_shape
        output_shape = model.output_shape
        print(f"📐 Model giriş boyutu: {input_shape}")
        print(f"📐 Model çıkış boyutu: {output_shape}")
        
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        exit(1)
    
    print("\n🎯 Test görselleri seçiliyor...")
    
    # Rastgele görseller seç
    with_mask_files = [f for f in os.listdir(with_mask_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    without_mask_files = [f for f in os.listdir(without_mask_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 3'er tane seç
    test_with_mask = random.sample(with_mask_files, 3)
    test_without_mask = random.sample(without_mask_files, 3)
    
    print(f"Seçilen maskeli görseller: {test_with_mask}")
    print(f"Seçilen maskesiz görseller: {test_without_mask}")
    
    print("\n🔍 Model testleri yapılıyor...")
    
    # Görselleştirme oluştur
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Yüz Maskesi Tespit - Model Sonuçları (Düzeltilmiş)', 
                fontsize=16, fontweight='bold')
    
    correct_predictions = 0
    total_predictions = 0
    
    # Test 1: Maskeli görseller
    print("Maskeli görseller test ediliyor:")
    for i, img_name in enumerate(test_with_mask):
        try:
            img_path = os.path.join(with_mask_dir, img_name)
            print(f"  Görsel {i+1}: {img_name}")
            
            # Görseli oku
            image = cv2.imread(img_path)
            if image is None:
                print(f"    ❌ Görsel okunamadı")
                continue
            
            # RGB'ye çevir
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Model için hazırla
            resized = cv2.resize(image_rgb, (224, 224))
            normalized = resized.astype('float32') / 255.0
            processed = np.expand_dims(normalized, axis=0)
            
            # Tahmin yap
            raw_prediction = model.predict(processed, verbose=0)[0][0]
            
            # DÜZELTME: Ters yorumlama
            if raw_prediction > 0.5:
                pred_label = "Maskesiz"
                confidence = raw_prediction * 100
                is_correct = False  # Yanlış tahmin
            else:
                pred_label = "Maskeli"
                confidence = (1 - raw_prediction) * 100
                is_correct = True   # Doğru tahmin
            
            total_predictions += 1
            if is_correct:
                correct_predictions += 1
            
            print(f"    Ham değer: {raw_prediction:.4f}")
            print(f"    Tahmin: {pred_label} (%{confidence:.1f})")
            print(f"    Sonuç: {'✓ Doğru' if is_correct else '✗ Yanlış'}")
            
            # Görselleştirme
            color = 'green' if is_correct else 'red'
            status = '✓' if is_correct else '✗'
            
            axes[0, i].imshow(image_rgb)
            axes[0, i].set_title(f'{status} Gerçek: Maskeli\nTahmin: {pred_label}\nGüven: %{confidence:.1f}',
                               color=color, fontweight='bold', fontsize=10)
            axes[0, i].axis('off')
            
        except Exception as e:
            print(f"    ❌ Hata: {e}")
    
    # Test 2: Maskesiz görseller  
    print("\nMaskesiz görseller test ediliyor:")
    for i, img_name in enumerate(test_without_mask):
        try:
            img_path = os.path.join(without_mask_dir, img_name)
            print(f"  Görsel {i+1}: {img_name}")
            
            # Görseli oku
            image = cv2.imread(img_path)
            if image is None:
                print(f"    ❌ Görsel okunamadı")
                continue
            
            # RGB'ye çevir
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Model için hazırla
            resized = cv2.resize(image_rgb, (224, 224))
            normalized = resized.astype('float32') / 255.0
            processed = np.expand_dims(normalized, axis=0)
            
            # Tahmin yap
            raw_prediction = model.predict(processed, verbose=0)[0][0]
            
            # DÜZELTME: Ters yorumlama
            if raw_prediction > 0.5:
                pred_label = "Maskesiz"
                confidence = raw_prediction * 100
                is_correct = True   # Doğru tahmin
            else:
                pred_label = "Maskeli"
                confidence = (1 - raw_prediction) * 100
                is_correct = False  # Yanlış tahmin
            
            total_predictions += 1
            if is_correct:
                correct_predictions += 1
            
            print(f"    Ham değer: {raw_prediction:.4f}")
            print(f"    Tahmin: {pred_label} (%{confidence:.1f})")
            print(f"    Sonuç: {'✓ Doğru' if is_correct else '✗ Yanlış'}")
            
            # Görselleştirme
            color = 'green' if is_correct else 'red'
            status = '✓' if is_correct else '✗'
            
            axes[1, i].imshow(image_rgb)
            axes[1, i].set_title(f'{status} Gerçek: Maskesiz\nTahmin: {pred_label}\nGüven: %{confidence:.1f}',
                               color=color, fontweight='bold', fontsize=10)
            axes[1, i].axis('off')
            
        except Exception as e:
            print(f"    ❌ Hata: {e}")
    
    # Sonuçları hesapla ve göster
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\n📊 SONUÇLAR:")
        print(f"   Toplam test: {total_predictions}")
        print(f"   Doğru tahmin: {correct_predictions}")
        print(f"   Doğruluk oranı: %{accuracy:.1f}")
        
        # Görsele doğruluk bilgisi ekle
        fig.text(0.5, 0.02, f'Doğruluk: {correct_predictions}/{total_predictions} (%{accuracy:.1f})', 
                 ha='center', fontsize=14, fontweight='bold')
    
    # Görseli kaydet
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.07)
    
    output_file = os.path.join(BASE_DIR, 'final_test_results.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Sonuç görseli kaydedildi: final_test_results.png")
    
    # Dosya bilgisi
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"📁 Dosya boyutu: {file_size:,} bytes")
    
    print("🎉 Test başarıyla tamamlandı!")
    print("🖼️  Görsel dosyasını açarak sonuçları inceleyebilirsiniz.")
    
except Exception as e:
    print(f"❌ Genel hata: {e}")
    import traceback
    traceback.print_exc()
