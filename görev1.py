import cv2
import numpy as np
import os

def process_hough_lines(image_path="resim.jpg"):
    # 1. Görseli yükle
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: '{image_path}' dosyası bulunamadı! 'resim.jpg' dosyasının kodla aynı klasörde olduğundan emin ol.")
        return

    # Kopyalar (Yeşil ve Kırmızı çizgiler çakışmasın diye ayrı görsellere çizilecek)
    img_standard = img.copy()
    img_probabilistic = img.copy()

    # Gri tonlamaya çevir
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- ADIM 1: Canny ile Kenar Tespiti ---
    # Yüksek eşiklerle arka plandaki ağaç/çim gürültülerini siliyoruz
    canny_edges = cv2.Canny(gray, threshold1=100, threshold2=200)

    # --- ADIM 2: Kenar Resmine Median Filtre Uygulama ---
    # ksize=3 kenarları silmeden ince parazit noktaları temizler
    smooth_edges = cv2.medianBlur(canny_edges, ksize=3)

    # --- ADIM 3A: Standart Hough Transform (cv2.HoughLines) ---
    # Yüksek threshold (160) sayesinde sadece ana güçlü doğrular çizilir
    lines_standard = cv2.HoughLines(smooth_edges, rho=1, theta=np.pi/180, threshold=160)

    if lines_standard is not None:
        for line in lines_standard:
            rho, theta = line.ravel()[:2]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            # Standart Hough çizgilerini YEŞİL (BGR: 0, 255, 0) renkle çizin
            cv2.line(img_standard, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # --- ADIM 3B: Olasılıksal Hough Transform (cv2.HoughLinesP) ---
    # minLineLength=80 ile kısa ıvır zıvır çizgileri eleyip şeritleri yakalıyoruz
    lines_p = cv2.HoughLinesP(smooth_edges, rho=1, theta=np.pi/180, threshold=80, minLineLength=80, maxLineGap=15)

    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line.ravel()
            # Olasılıksal Hough çizgilerini KIRMIZI (BGR: 0, 0, 255) renkle çizin
            cv2.line(img_probabilistic, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # --- ADIM 4: Çıktıları Kaydetme ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cv2.imwrite(os.path.join(script_dir, '1_canny_edges.jpg'), canny_edges)
    cv2.imwrite(os.path.join(script_dir, '2_smooth_edges.jpg'), smooth_edges)
    cv2.imwrite(os.path.join(script_dir, '3_standard_hough.jpg'), img_standard)
    cv2.imwrite(os.path.join(script_dir, '4_probabilistic_hough.jpg'), img_probabilistic)

    print("İşlem başarıyla tamamlandı! Çıktılar 6.hafta klasörüne kaydoldu.")

    # Ekran Pencereleri
    cv2.imshow('1 - Canny Kenarlar', canny_edges)
    cv2.imshow('2 - Median Filtreli Kenarlar', smooth_edges)
    cv2.imshow('3 - Standart Hough (Yesil Cizgiler)', img_standard)
    cv2.imshow('4 - Olasiliksal Hough (Kirmizi Cizgiler)', img_probabilistic)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_full_path = os.path.join(script_dir, 'resim.jpg')
    
    process_hough_lines(image_full_path)