"""
UI Components for OCR Image Viewer Application
Contains reusable UI widgets and components
"""

import os
from collections import deque

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QListWidget, 
    QListWidgetItem, QTextEdit, QGridLayout, QScrollArea
)
from PySide6.QtGui import QPixmap, QIcon, QImage
from PySide6.QtCore import Qt, QSize, QEvent, QMimeData, Signal


class ClickableLabel(QLabel):
    """Clickable label widget that emits a signal when clicked"""
    clicked = Signal(int)  # Send index when clicked

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)


class PreviewGridWidget(QWidget):
    """Grid widget for displaying preview images"""
    imageClicked = Signal(int)  # Signal to pass to external

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        self.image_labels = []

    def clear(self):
        """Clear all image labels from the grid"""
        for label in self.image_labels:
            self.grid_layout.removeWidget(label)
            label.deleteLater()
        self.image_labels.clear()

    def set_images(self, image_q: deque):
        """Set images to display in the grid"""
        self.clear()

        cols = 2  # Number of desired columns
        for i in range(len(image_q)):
            idx = len(image_q) - i - 1
            label = ClickableLabel(idx)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("border: 1px solid gray; background-color: white;")
            label.clicked.connect(self.imageClicked.emit)

            # Display if image is in numpy array format
            rgb = image_q[idx][0]
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)

            # Scale to fit label size
            pixmap = pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(pixmap)

            row = i // cols
            col = i % cols
            self.grid_layout.addWidget(label, row, col)
            self.image_labels.append(label)


class ThumbnailListWidget(QListWidget):
    """List widget for displaying image thumbnails with drag and drop support"""
    
    def __init__(self, image_click_callback, thumbnail_size=60):
        super().__init__()
        self.setViewMode(QListWidget.ViewMode.IconMode)
        # Set smaller fixed square size for thumbnails
        self.setIconSize(QSize(thumbnail_size, thumbnail_size))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        # Reduce spacing for more compact layout
        self.setSpacing(5)
        self.image_click_callback = image_click_callback
        self.itemClicked.connect(self.handle_item_click)
        self.thumbnail_size = thumbnail_size

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if os.path.isfile(file_path) and self.is_image(file_path):
                    self.add_thumbnail(file_path)
        event.acceptProposedAction()

    def is_image(self, path):
        """Check if the file is an image based on extension"""
        return path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

    def truncate_filename(self, filename, max_length):
        """
        Truncate filename to fit within thumbnail width
        """
        if len(filename) <= max_length:
            return filename
        
        # Calculate available space for filename (considering padding and ellipsis)
        available_chars = max_length - 3  # Reserve 3 characters for "..."
        
        if available_chars <= 0:
            return "..."
        
        # Split filename and extension
        name, ext = os.path.splitext(filename)
        
        # If extension is too long, truncate it
        if len(ext) > available_chars // 2:
            ext = ext[:available_chars // 2]
        
        # Calculate remaining space for name
        name_chars = available_chars - len(ext)
        
        if name_chars <= 0:
            return "..." + ext
        
        # Truncate name and add ellipsis
        truncated_name = name[:name_chars] + "..."
        return truncated_name + ext

    def add_thumbnail(self, file_path):
        """Add a thumbnail to the list"""
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            # Create square thumbnail with fixed size
            icon = QIcon(pixmap.scaled(
                self.thumbnail_size, 
                self.thumbnail_size, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))
            
            # Get filename and truncate it to fit thumbnail width
            filename = os.path.basename(file_path)
            # Estimate max characters that can fit in thumbnail width
            # Assuming average character width of 6-8 pixels and some padding
            max_chars = max(8, self.thumbnail_size // 8)
            truncated_filename = self.truncate_filename(filename, max_chars)
            
            # Show truncated filename
            item = QListWidgetItem(icon, truncated_filename)
            item.setData(Qt.ItemDataRole.UserRole, file_path)
            # Store full filename as tooltip for hover display
            item.setToolTip(filename)
            self.addItem(item)

    def handle_item_click(self, item):
        """Handle item click event"""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        self.image_click_callback(file_path)


class PerformanceInfoWidget(QWidget):
    """Widget for displaying performance information"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Performance Information")
        title.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: black;
                padding: 2px;
                border-radius: 2px;
                margin-bottom: 3px;
            }
        """)
        layout.addWidget(title)
        
        # Performance table
        self.performance_table = QTextEdit()
        self.performance_table.setReadOnly(True)
        self.performance_table.setMaximumHeight(120)
        self.performance_table.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 10px;
                line-height: 1.2;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 3px;
                padding: 2px;
            }
        """)
        
        # Set initial performance data
        self.update_performance_data()
        
        layout.addWidget(self.performance_table)
        self.setLayout(layout)
    
    def update_performance_data(self, det_gpu=10, det_npu=10.548, cls_gpu=95.3, cls_npu=0.19, rec_gpu=121, rec_npu=0.736):
        """
        Update performance information display
        """
        performance_text = f"""     | GPU | NPU
det | {det_gpu:.2f} | {1000/(det_npu + 1e-10):.2f}
cls | {cls_gpu:.2f} | {1000/(cls_npu + 1e-10):.2f}
rec | {rec_gpu:.2f} | {1000/(rec_npu + 1e-10):.2f}(max)"""
        
        self.performance_table.setText(performance_text) 