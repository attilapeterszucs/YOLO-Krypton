"""
YOLO Krypton - UI Components Module
Reusable UI components for YOLO Krypton application
"""

import customtkinter as ctk
from typing import Dict, Any, Optional
import tkinter as tk


def create_stat_card(parent, title: str, value: str, icon: str) -> ctk.CTkFrame:
    """Create a statistics card widget"""
    card = ctk.CTkFrame(parent, height=100)
    
    icon_label = ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=24))
    icon_label.pack(pady=(10, 5))
    
    value_label = ctk.CTkLabel(
        card,
        text=value,
        font=ctk.CTkFont(size=20, weight="bold")
    )
    value_label.pack()
    
    title_label = ctk.CTkLabel(
        card,
        text=title,
        font=ctk.CTkFont(size=12),
        text_color="gray"
    )
    title_label.pack(pady=(5, 10))
    
    # Store value label for updates
    card.value_label = value_label
    
    return card


def update_stat_card(card: ctk.CTkFrame, value: str):
    """Update the value of a stat card"""
    if hasattr(card, 'value_label'):
        card.value_label.configure(text=value)


def create_results_display(parent) -> ctk.CTkTextbox:
    """Create a results display text widget"""
    results_text = ctk.CTkTextbox(
        parent,
        height=150,
        font=ctk.CTkFont(family="Consolas", size=11)
    )
    return results_text


def update_results_display(results_text: ctk.CTkTextbox, results: Dict[str, Any]):
    """Update the results text display"""
    results_text.delete("1.0", "end")
    
    if results.get('detections'):
        text = f"Total Objects Detected: {results.get('total_objects', 0)}\n"
        text += "-" * 50 + "\n"
        
        for i, det in enumerate(results['detections'][:20], 1):  # Show first 20
            text += f"{i}. {det['class_name']}: {det['confidence']:.2%}\n"
            text += f"   Location: [{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, "
            text += f"{det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]\n"
        
        if len(results['detections']) > 20:
            text += f"\n... and {len(results['detections']) - 20} more objects"
    else:
        text = "No objects detected"
    
    results_text.insert("1.0", text)


class DetectionPanel(ctk.CTkFrame):
    """Panel for displaying detection results"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the detection panel UI"""
        # Title
        title = ctk.CTkLabel(
            self,
            text="Detection Results",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.pack(anchor="w", padx=10, pady=10)
        
        # Results area
        self.results_frame = ctk.CTkScrollableFrame(self)
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.result_items = []
    
    def add_detection(self, class_name: str, confidence: float, bbox: list):
        """Add a detection result to the panel"""
        item_frame = ctk.CTkFrame(self.results_frame)
        item_frame.pack(fill="x", pady=2)
        
        # Class name and confidence
        info_label = ctk.CTkLabel(
            item_frame,
            text=f"{class_name}: {confidence:.2%}",
            font=ctk.CTkFont(size=12)
        )
        info_label.pack(side="left", padx=10)
        
        # Bounding box info
        bbox_label = ctk.CTkLabel(
            item_frame,
            text=f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        bbox_label.pack(side="right", padx=10)
        
        self.result_items.append(item_frame)
    
    def clear_detections(self):
        """Clear all detection results"""
        for item in self.result_items:
            item.destroy()
        self.result_items = []


class SidePanel(ctk.CTkFrame):
    """Sidebar panel with controls"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, width=300, corner_radius=0, **kwargs)
        
        self.grid_rowconfigure(4, weight=1)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the sidebar UI"""
        # Logo
        logo = ctk.CTkLabel(
            self,
            text="🎯 YOLO Studio",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        logo.grid(row=0, column=0, padx=20, pady=(20, 10))


class StatusBar(ctk.CTkFrame):
    """Status bar for showing application status"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, height=30, **kwargs)
        
        self.status_label = ctk.CTkLabel(
            self,
            text="Ready",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="left", padx=10)
        
        self.progress_bar = ctk.CTkProgressBar(self, width=200)
        self.progress_bar.pack(side="right", padx=10)
        self.progress_bar.set(0)
    
    def update_status(self, message: str):
        """Update status message"""
        self.status_label.configure(text=message)
    
    def set_progress(self, value: float):
        """Set progress bar value (0-1)"""
        self.progress_bar.set(value)


class VideoPlayer(ctk.CTkFrame):
    """Video player widget for displaying video frames"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.canvas = ctk.CTkLabel(
            self,
            text="No video loaded",
            font=ctk.CTkFont(size=14)
        )
        self.canvas.pack(fill="both", expand=True)
        
        # Control buttons
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.pack(fill="x", pady=10)
        
        self.play_btn = ctk.CTkButton(
            self.controls_frame,
            text="▶ Play",
            width=80
        )
        self.play_btn.pack(side="left", padx=5)
        
        self.pause_btn = ctk.CTkButton(
            self.controls_frame,
            text="⏸ Pause",
            width=80
        )
        self.pause_btn.pack(side="left", padx=5)
        
        self.stop_btn = ctk.CTkButton(
            self.controls_frame,
            text="⏹ Stop",
            width=80
        )
        self.stop_btn.pack(side="left", padx=5)
        
        # Progress slider
        self.progress_slider = ctk.CTkSlider(
            self.controls_frame,
            from_=0,
            to=100
        )
        self.progress_slider.pack(side="left", fill="x", expand=True, padx=10)
    
    def update_frame(self, image):
        """Update the displayed frame"""
        self.canvas.configure(image=image, text="")
        self.canvas.image = image


class ResultsPanel(ctk.CTkFrame):
    """Panel for displaying detailed results and statistics"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the results panel UI"""
        # Create notebook for tabs
        self.notebook = ctk.CTkTabview(self)
        self.notebook.pack(fill="both", expand=True)
        
        # Summary tab
        self.notebook.add("Summary")
        self.summary_tab = self.notebook.tab("Summary")
        
        # Details tab
        self.notebook.add("Details")
        self.details_tab = self.notebook.tab("Details")
        
        # Export tab
        self.notebook.add("Export")
        self.export_tab = self.notebook.tab("Export")
        
        self._setup_summary_tab()
        self._setup_details_tab()
        self._setup_export_tab()
    
    def _setup_summary_tab(self):
        """Setup the summary tab"""
        summary_text = ctk.CTkTextbox(
            self.summary_tab,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        summary_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.summary_text = summary_text
    
    def _setup_details_tab(self):
        """Setup the details tab"""
        details_frame = ctk.CTkScrollableFrame(self.details_tab)
        details_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.details_frame = details_frame
    
    def _setup_export_tab(self):
        """Setup the export tab"""
        export_frame = ctk.CTkFrame(self.export_tab)
        export_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Export format selection
        format_label = ctk.CTkLabel(
            export_frame,
            text="Export Format:",
            font=ctk.CTkFont(size=14)
        )
        format_label.pack(anchor="w", pady=10)
        
        self.format_var = ctk.StringVar(value="JSON")
        format_menu = ctk.CTkOptionMenu(
            export_frame,
            values=["JSON", "CSV", "TXT", "YOLO"],
            variable=self.format_var
        )
        format_menu.pack(fill="x", pady=5)
        
        # Export button
        export_btn = ctk.CTkButton(
            export_frame,
            text="Export Results",
            height=40
        )
        export_btn.pack(pady=20)


class StatisticsView(ctk.CTkFrame):
    """View for displaying detection statistics"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the statistics view UI"""
        # Title
        title = ctk.CTkLabel(
            self,
            text="Detection Statistics",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(pady=10)
        
        # Stats grid
        stats_frame = ctk.CTkFrame(self)
        stats_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Create stat cards
        self.stat_cards = {}
        
        # Total detections
        self.stat_cards['total'] = self._create_stat_item(
            stats_frame,
            "Total Detections",
            "0"
        )
        
        # Unique classes
        self.stat_cards['classes'] = self._create_stat_item(
            stats_frame,
            "Unique Classes",
            "0"
        )
        
        # Average confidence
        self.stat_cards['confidence'] = self._create_stat_item(
            stats_frame,
            "Average Confidence",
            "0%"
        )
        
        # Processing time
        self.stat_cards['time'] = self._create_stat_item(
            stats_frame,
            "Processing Time",
            "0ms"
        )
        
        # Arrange in grid
        row = 0
        col = 0
        for card in self.stat_cards.values():
            card.grid(row=row, column=col, padx=10, pady=10, sticky="ew")
            col += 1
            if col > 1:
                col = 0
                row += 1
        
        # Configure grid weights
        for i in range(2):
            stats_frame.grid_columnconfigure(i, weight=1)
    
    def _create_stat_item(self, parent, label: str, value: str) -> ctk.CTkFrame:
        """Create a statistics item"""
        frame = ctk.CTkFrame(parent)
        
        label_widget = ctk.CTkLabel(
            frame,
            text=label,
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        label_widget.pack(pady=(10, 5))
        
        value_widget = ctk.CTkLabel(
            frame,
            text=value,
            font=ctk.CTkFont(size=20, weight="bold")
        )
        value_widget.pack(pady=(0, 10))
        
        frame.value_label = value_widget
        return frame
    
    def update_stat(self, stat_name: str, value: str):
        """Update a statistic value"""
        if stat_name in self.stat_cards:
            self.stat_cards[stat_name].value_label.configure(text=value)
