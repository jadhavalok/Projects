from tkinter import filedialog
import tkinter as tk
from tkinter import ttk, scrolledtext, StringVar, messagebox
from ttkbootstrap import Style
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from pytesseract import image_to_string
from PIL import Image

class PDFSummarizerApp:
    def __init__(self, master):
        self.master = master
        master.title("PDF Summarizer")

        # Default theme is light mode
        self.dark_mode = False

        # Create and pack widgets
        self.create_widgets()

    def create_widgets(self):
        # Apply a Bootstrap-themed style with light or dark mode
        theme = 'darkly' if self.dark_mode else 'flatly'
        style = Style(theme=theme)

        self.text_area = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, width=40, height=10)
        self.text_area.pack(padx=10, pady=10)

        self.browse_button = ttk.Button(self.master, text="Browse PDF", command=self.load_pdf)
        self.browse_button.pack(pady=5)

        self.summarize_button = ttk.Button(self.master, text="Summarize", command=self.summarize_content)
        self.summarize_button.pack(pady=5)

        self.custom_ratio_label = ttk.Label(self.master, text="Custom Summarization Ratio:")
        self.custom_ratio_label.pack(pady=5)

        self.custom_ratio_var = StringVar()
        self.custom_ratio_entry = ttk.Entry(self.master, textvariable=self.custom_ratio_var)
        self.custom_ratio_entry.pack(pady=5)

        self.summary_label = ttk.Label(self.master, text="Summary:")
        self.summary_label.pack(pady=5)

        self.summary_area = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, width=40, height=10)
        self.summary_area.pack(padx=10, pady=10)

        self.toggle_mode_button = ttk.Button(self.master, text="Toggle Dark Mode", command=self.toggle_dark_mode)
        self.toggle_mode_button.pack(pady=5)

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        # Destroy the current widgets
        for widget in self.master.winfo_children():
            widget.destroy()
        # Create new widgets with the updated dark mode setting
        self.create_widgets()

    def load_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        pdf_text = self.extract_text_from_pdf(file_path)
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, pdf_text)

    def extract_text_from_pdf(self, pdf_path):
        # Extract text from PDF using pytesseract
        images = convert_pdf_to_images(pdf_path)
        text = ""
        for img in images:
            text += image_to_string(img) + " "
        return text

    def summarizze_content(self):
        pdf_text = self.text_area.get("1.0", tk.END)
        custom_ratio_str = self.custom_ratio_var.get()

        if not custom_ratio_str:
            error_message = "Please enter a custom summarization ratio."
            messagebox.showerror("Error", error_message)
            return

        try:
            custom_ratio = float(custom_ratio_str)
            summary = self.generate_summary(pdf_text, num_sentences=int(custom_ratio))
            self.summary_area.delete('1.0', tk.END)
            self.summary_area.insert(tk.END, summary)
        except ValueError:
            error_message = "Invalid input. Please enter a valid number for the custom summarization ratio."
            messagebox.showerror("Error", error_message)

    def generate_summary(self, text, num_sentences=5):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = ""
        for sentence in summarizer(parser.document, num_sentences):
            summary += str(sentence) + " "
        return summary

# Function to convert PDF to images
def convert_pdf_to_images(pdf_path):
    images = []
    pdf_file = Image.open(pdf_path)
    for page_num in range(pdf_file.n_pages):
        pdf_file.seek(page_num)
        page = pdf_file.load()
        image = Image.new("RGB", pdf_file.size, (255, 255, 255))
        image.paste(page, (0, 0))
        images.append(image)
    return images

# Main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = PDFSummarizerApp(root)
    root.mainloop()
