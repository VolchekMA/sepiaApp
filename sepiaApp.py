import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk


class Processor:
    """Загрузка изображения и его обработка (сепия, виньетка)."""

    # Матрица коэффициентов сепии для каналов B, G, R
    SEPIA_MATRIX = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189],
    ])

    def __init__(self):
        self.original = None

    def load(self, path):
        """Загружает изображение с диска с помощью cv2.imread."""
        image = cv2.imread(path)
        if image is None:
            raise ValueError("Не удалось загрузить изображение")
        self.original = image
        return self.original

    def apply_sepia(self, img):
        """
        Умножает каналы матрицы изображения на коэффициенты сепии
        с помощью cv2.transform и ограничивает значения до 255.
        """
        sepia_img = cv2.transform(img, self.SEPIA_MATRIX)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return sepia_img

    def apply_vignette(self, img, radius):
        """
        Накладывает маску виньетки: для каждого пикселя считается
        расстояние от центра изображения, и края плавно затемняются
        пропорционально значению radius (50 — сильное затемнение,
        300 — почти без затемнения).
        """
        h, w = img.shape[:2]
        cx, cy = w / 2, h / 2

        y_idx, x_idx = np.indices((h, w))
        dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)

        strength = radius / 300.0
        mask = 1 - (dist / max_dist) * (1 - strength)
        mask = np.clip(mask, 0, 1)[..., np.newaxis]

        result = img.astype(np.float32) * mask
        return np.clip(result, 0, 255).astype(np.uint8)

    def process(self, sepia_enabled, radius):
        """Конвейер обработки: сепия (опционально) + виньетка (всегда)."""
        if self.original is None:
            raise ValueError("Изображение не загружено")

        img = self.original.copy()
        if sepia_enabled:
            img = self.apply_sepia(img)
        img = self.apply_vignette(img, radius)
        return img

    def save(self, img, path):
        """Сохраняет результат обработки в файл (JPEG/PNG)."""
        cv2.imwrite(path, img)


class App(tk.Tk):
    """Главное окно приложения с двумя вкладками."""

    def __init__(self):
        super().__init__()
        self.title("Стилизация фотографии под старину")
        self.geometry("650x650")

        self.processor = Processor()
        self.result_image = None

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.tab_settings = ttk.Frame(notebook)
        self.tab_result = ttk.Frame(notebook)
        notebook.add(self.tab_settings, text="Настройки")
        notebook.add(self.tab_result, text="Результат")

        self._build_settings_tab()
        self._build_result_tab()

    def _build_settings_tab(self):
        tab = self.tab_settings

        btn_load = tk.Button(tab, text="Загрузить изображение", command=self.load_image)
        btn_load.grid(column=0, row=0, columnspan=2, padx=10, pady=10, sticky="w")

        self.sepia_var = tk.BooleanVar(value=False)
        chk_sepia = tk.Checkbutton(tab, text="Включить сепию", variable=self.sepia_var)
        chk_sepia.grid(column=0, row=1, columnspan=2, padx=10, pady=5, sticky="w")

        lbl_radius = tk.Label(tab, text="Радиус виньетки:")
        lbl_radius.grid(column=0, row=2, padx=10, pady=5, sticky="w")

        self.radius_var = tk.IntVar(value=300)
        scale_radius = tk.Scale(tab, from_=50, to=300, orient="horizontal",
                                 variable=self.radius_var, length=250)
        scale_radius.grid(column=1, row=2, padx=10, pady=5, sticky="w")

        btn_process = tk.Button(tab, text="Применить эффекты", command=self.process_image)
        btn_process.grid(column=0, row=3, columnspan=2, padx=10, pady=10, sticky="w")

        self.label_original = tk.Label(tab, text="Изображение не загружено")
        self.label_original.grid(column=0, row=4, columnspan=2, padx=10, pady=10)

    def _build_result_tab(self):
        tab = self.tab_result

        self.label_result = tk.Label(tab, text="Результат появится здесь")
        self.label_result.pack(padx=10, pady=10)

        btn_save = tk.Button(tab, text="Сохранить результат (JPEG/PNG)", command=self.save_image)
        btn_save.pack(padx=10, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return
        try:
            img = self.processor.load(path)
        except ValueError as err:
            messagebox.showerror("Ошибка", str(err))
            return
        self._show_image(img, self.label_original)

    def process_image(self):
        if self.processor.original is None:
            messagebox.showwarning("Внимание", "Сначала загрузите изображение")
            return

        sepia_enabled = self.sepia_var.get()
        radius = self.radius_var.get()
        self.result_image = self.processor.process(sepia_enabled, radius)
        self._show_image(self.result_image, self.label_result)

    def save_image(self):
        if self.result_image is None:
            messagebox.showwarning("Внимание", "Нет результата для сохранения")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if not path:
            return
        self.processor.save(self.result_image, path)
        messagebox.showinfo("Готово", f"Изображение сохранено: {path}")

    def _show_image(self, cv_img, label_widget):
        """Отображает cv2-изображение (BGR) в виджете Label."""
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((500, 500))
        tk_img = ImageTk.PhotoImage(pil_img)
        label_widget.configure(image=tk_img, text="")
        label_widget.image = tk_img


if __name__ == "__main__":
    app = App()
    app.mainloop()
