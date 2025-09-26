"""YOLO Krypton - Professional Object Detection Application
A modern, camera-first application for real-time object detection using YOLOv8
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox, Menu
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
import threading
import time
from datetime import datetime
import json
from typing import Optional, Dict, Any
import config
from yolo_detector import YOLODetector
import os
from ui_components import (
    create_stat_card, update_stat_card,
    create_results_display, update_results_display
)

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class YOLODetectionApp(ctk.CTk):
    """Main application class for YOLO Object Detection Studio"""
    
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title(config.APP_NAME)
        self.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.minsize(1200, 700)
        
        # Center window on screen
        self._center_window()
        
        # Initialize variables
        self.detector = None
        self.current_image = None
        self.current_video = "webcam"  # Start with webcam by default
        self.current_video_path = None
        self.detection_results = []
        self.is_detecting = False
        self.video_thread = None
        self.stop_video = threading.Event()
        self.camera_auto_start = True
        self.current_fps = 0
        self.frame_skip = 0
        self.video_paused = False
        self.video_frame_count = 0
        self.video_total_frames = 0
        self.video_fps = 30
        self.video_speed = 1.0
        self.video_seeking = False
        
        # Initialize UI
        self._setup_ui()
        self._setup_menu_bar()
        self._load_initial_model()
        
        # Bind keyboard shortcuts
        self._setup_shortcuts()
        
        # Start camera by default after model loads
        self.after(2000, self._start_default_camera)
        
    def _center_window(self):
        """Center the window on screen"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
    
    def _setup_ui(self):
        """Setup the main UI layout"""
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create main containers
        self._create_sidebar()
        self._create_main_content()
        self._create_status_bar()
    
    def _setup_menu_bar(self):
        """Setup the menu bar"""
        # Create menu bar
        menubar = Menu(self, bg='#2d2d2d', fg='white', activebackground='#1e88e5', activeforeground='white')
        self.config(menu=menubar)
        
        # File menu
        file_menu = Menu(menubar, tearoff=0, bg='#2d2d2d', fg='white')
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="📷 Open Image...", command=self._load_image, accelerator="Ctrl+O")
        file_menu.add_command(label="🎥 Open Video...", command=self._load_video, accelerator="Ctrl+V")
        file_menu.add_command(label="📹 Use Camera", command=self._switch_to_camera, accelerator="Ctrl+W")
        file_menu.add_separator()
        file_menu.add_command(label="💾 Export Results...", command=self._export_results, accelerator="Ctrl+E")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        # View menu
        view_menu = Menu(menubar, tearoff=0, bg='#2d2d2d', fg='white')
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Detection View", command=lambda: self.tabview.set("Detection View"))
        view_menu.add_command(label="Statistics", command=lambda: self.tabview.set("Statistics"))
        
        # Model menu
        model_menu = Menu(menubar, tearoff=0, bg='#2d2d2d', fg='white')
        menubar.add_cascade(label="Model", menu=model_menu)
        for model_name in config.AVAILABLE_MODELS.keys():
            model_menu.add_radiobutton(
                label=model_name,
                command=lambda m=model_name: self._change_model_from_menu(m)
            )
        
        # Help menu
        help_menu = Menu(menubar, tearoff=0, bg='#2d2d2d', fg='white')
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)
    
    def _create_sidebar(self):
        """Create the sidebar with controls"""
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)
        
        # Logo and title
        logo_label = ctk.CTkLabel(
            self.sidebar,
            text="🚀 YOLO Krypton",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Camera status indicator
        self.camera_status = ctk.CTkLabel(
            self.sidebar,
            text="● Camera Active",
            font=ctk.CTkFont(size=12),
            text_color="#43a047"
        )
        self.camera_status.grid(row=0, column=0, padx=20, pady=(55, 0))
        
        # Model selection
        model_frame = ctk.CTkFrame(self.sidebar)
        model_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            model_frame,
            text="Model Selection",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        self.model_var = ctk.StringVar(value="YOLOv8n (Nano - Fastest)")
        self.model_dropdown = ctk.CTkComboBox(
            model_frame,
            values=list(config.AVAILABLE_MODELS.keys()),
            variable=self.model_var,
            command=self._on_model_change,
            width=250
        )
        self.model_dropdown.pack(padx=10, pady=(0, 10))
        
        # Detection parameters
        params_frame = ctk.CTkFrame(self.sidebar)
        params_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            params_frame,
            text="Detection Parameters",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Confidence threshold
        ctk.CTkLabel(params_frame, text="Confidence Threshold:").pack(
            anchor="w", padx=10, pady=(5, 0)
        )
        
        self.confidence_slider = ctk.CTkSlider(
            params_frame,
            from_=0.1,
            to=1.0,
            number_of_steps=90,
            command=self._update_confidence_label
        )
        self.confidence_slider.set(config.DEFAULT_CONFIDENCE)
        self.confidence_slider.pack(padx=10, pady=5, fill="x")
        
        self.confidence_label = ctk.CTkLabel(
            params_frame,
            text=f"{config.DEFAULT_CONFIDENCE:.2f}"
        )
        self.confidence_label.pack(padx=10)
        
        # IOU threshold
        ctk.CTkLabel(params_frame, text="IOU Threshold:").pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        
        self.iou_slider = ctk.CTkSlider(
            params_frame,
            from_=0.1,
            to=1.0,
            number_of_steps=90,
            command=self._update_iou_label
        )
        self.iou_slider.set(config.DEFAULT_IOU_THRESHOLD)
        self.iou_slider.pack(padx=10, pady=5, fill="x")
        
        self.iou_label = ctk.CTkLabel(
            params_frame,
            text=f"{config.DEFAULT_IOU_THRESHOLD:.2f}"
        )
        self.iou_label.pack(padx=10)
        
        # Device selection
        device_frame = ctk.CTkFrame(self.sidebar)
        device_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            device_frame,
            text="Processing Device",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        self.device_var = ctk.StringVar(value=config.DEFAULT_DEVICE)
        self.device_dropdown = ctk.CTkComboBox(
            device_frame,
            values=config.DEVICE_OPTIONS,
            variable=self.device_var,
            command=self._on_device_change,
            width=250
        )
        self.device_dropdown.pack(padx=10, pady=(0, 10))
        
        # Device info label
        self.device_info_label = ctk.CTkLabel(
            device_frame,
            text="Device: Detecting...",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.device_info_label.pack(anchor="w", padx=10, pady=(0, 5))
        
        # Performance settings
        perf_frame = ctk.CTkFrame(self.sidebar)
        perf_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            perf_frame,
            text="Performance",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Frame skip slider
        ctk.CTkLabel(perf_frame, text="Frame Skip (0=None):").pack(
            anchor="w", padx=10, pady=(5, 0)
        )
        
        self.frame_skip_slider = ctk.CTkSlider(
            perf_frame,
            from_=0,
            to=5,
            number_of_steps=5,
            command=self._update_frame_skip
        )
        self.frame_skip_slider.set(0)
        self.frame_skip_slider.pack(padx=10, pady=5, fill="x")
        
        self.frame_skip_label = ctk.CTkLabel(
            perf_frame,
            text="0 (Process all frames)"
        )
        self.frame_skip_label.pack(padx=10, pady=(0, 10))
        
        # FPS display
        self.fps_label = ctk.CTkLabel(
            perf_frame,
            text="FPS: 0.0",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#43a047"
        )
        self.fps_label.pack(anchor="w", padx=10, pady=(5, 10))
        
        # Camera controls
        camera_frame = ctk.CTkFrame(self.sidebar)
        camera_frame.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            camera_frame,
            text="Camera Controls",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        self.camera_btn = ctk.CTkButton(
            camera_frame,
            text="⏸ Pause Camera",
            command=self._toggle_camera,
            height=40,
            fg_color="orange"
        )
        self.camera_btn.pack(padx=10, pady=5, fill="x")
        
        self.snapshot_btn = ctk.CTkButton(
            camera_frame,
            text="📸 Take Snapshot",
            command=self._take_snapshot,
            height=40
        )
        self.snapshot_btn.pack(padx=10, pady=5, fill="x")
        
        # Input source indicator
        source_frame = ctk.CTkFrame(self.sidebar)
        source_frame.grid(row=6, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            source_frame,
            text="Current Source",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        self.source_label = ctk.CTkLabel(
            source_frame,
            text="📹 Live Camera",
            font=ctk.CTkFont(size=12)
        )
        self.source_label.pack(anchor="w", padx=10, pady=(0, 10))
        
        # Quick switch buttons
        switch_frame = ctk.CTkFrame(source_frame)
        switch_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.switch_camera_btn = ctk.CTkButton(
            switch_frame,
            text="📹",
            command=self._switch_to_camera,
            width=60,
            height=30
        )
        self.switch_camera_btn.pack(side="left", padx=2)
        
        self.switch_image_btn = ctk.CTkButton(
            switch_frame,
            text="📷",
            command=self._load_image,
            width=60,
            height=30
        )
        self.switch_image_btn.pack(side="left", padx=2)
        
        self.switch_video_btn = ctk.CTkButton(
            switch_frame,
            text="🎥",
            command=self._load_video,
            width=60,
            height=30
        )
        
        self.switch_video_btn.pack(side="left", padx=2)
        
        # Export section
        export_frame = ctk.CTkFrame(self.sidebar)
        export_frame.grid(row=7, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            export_frame,
            text="Export Results",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        self.export_format = ctk.StringVar(value="JSON")
        export_dropdown = ctk.CTkComboBox(
            export_frame,
            values=config.EXPORT_FORMATS,
            variable=self.export_format,
            width=250
        )
        export_dropdown.pack(padx=10, pady=5)
        
        self.export_btn = ctk.CTkButton(
            export_frame,
            text="💾 Export Results",
            command=self._export_results,
            height=35
        )
        self.export_btn.pack(padx=10, pady=5, fill="x")
    
    def _create_main_content(self):
        """Create the main content area"""
        self.main_container = ctk.CTkFrame(self)
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=3)
        self.main_container.grid_rowconfigure(1, weight=1)
        
        # Detection display area
        self.display_frame = ctk.CTkFrame(self.main_container)
        self.display_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        
        # Create tabview for different views
        self.tabview = ctk.CTkTabview(self.display_frame)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Detection tab
        self.tabview.add("Detection View")
        self.detection_tab = self.tabview.tab("Detection View")
        
        self.canvas = ctk.CTkLabel(
            self.detection_tab,
            text="Initializing camera...",
            font=ctk.CTkFont(size=16)
        )
        self.canvas.pack(fill="both", expand=True)
        
        # Statistics tab
        self.tabview.add("Statistics")
        self.stats_tab = self.tabview.tab("Statistics")
        self._create_statistics_view()
        
        # Results panel
        self.results_frame = ctk.CTkFrame(self.main_container)
        self.results_frame.grid(row=1, column=0, sticky="nsew")
        
        results_label = ctk.CTkLabel(
            self.results_frame,
            text="Detection Results",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        results_label.pack(anchor="w", padx=10, pady=5)
        
        # Results text area with scrollbar
        self.results_text = ctk.CTkTextbox(
            self.results_frame,
            height=150,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.results_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    def _create_video_controls(self):
        """Create video playback controls"""
        self.video_controls_frame = ctk.CTkFrame(self.detection_tab)
        
        # Video seek slider (acts as progress bar and seek control)
        seek_frame = ctk.CTkFrame(self.video_controls_frame)
        seek_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        self.video_slider = ctk.CTkSlider(
            seek_frame,
            from_=0,
            to=100,
            command=self._on_video_seek,
            width=None
        )
        self.video_slider.set(0)
        self.video_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        # Time display next to slider
        self.video_time_label = ctk.CTkLabel(
            seek_frame,
            text="0:00 / 0:00",
            font=ctk.CTkFont(size=11)
        )
        self.video_time_label.pack(side="right")
        
        # Main controls container
        controls_container = ctk.CTkFrame(self.video_controls_frame)
        controls_container.pack(fill="x", padx=10, pady=5)
        
        # Left controls (playback)
        left_controls = ctk.CTkFrame(controls_container)
        left_controls.pack(side="left")
        
        # Restart button
        self.video_restart_btn = ctk.CTkButton(
            left_controls,
            text="⏮",
            width=40,
            command=self._restart_video
        )
        self.video_restart_btn.pack(side="left", padx=2)
        
        # Play/Pause button
        self.video_play_btn = ctk.CTkButton(
            left_controls,
            text="⏸",
            width=40,
            command=self._toggle_video_playback
        )
        self.video_play_btn.pack(side="left", padx=2)
        
        # Stop button
        self.video_stop_btn = ctk.CTkButton(
            left_controls,
            text="⏹",
            width=40,
            command=self._stop_video_playback
        )
        self.video_stop_btn.pack(side="left", padx=2)
        
        # Center info
        center_info = ctk.CTkFrame(controls_container)
        center_info.pack(side="left", padx=20)
        
        self.video_info_label = ctk.CTkLabel(
            center_info,
            text="No video loaded",
            font=ctk.CTkFont(size=11)
        )
        self.video_info_label.pack()
        
        # Right controls (speed)
        right_controls = ctk.CTkFrame(controls_container)
        right_controls.pack(side="right", padx=10)
        
        ctk.CTkLabel(
            right_controls,
            text="Speed:",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(0, 5))
        
        self.video_speed_var = ctk.StringVar(value="1.0x")
        self.video_speed_menu = ctk.CTkComboBox(
            right_controls,
            values=["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"],
            variable=self.video_speed_var,
            command=self._on_speed_change,
            width=80
        )
        self.video_speed_menu.pack(side="left")
        
        # Loop checkbox
        self.video_loop_var = ctk.BooleanVar(value=False)
        self.video_loop_check = ctk.CTkCheckBox(
            right_controls,
            text="Loop",
            variable=self.video_loop_var,
            width=60
        )
        self.video_loop_check.pack(side="left", padx=(10, 0))
        
        # Initially hide video controls
        # They will be shown when a video is loaded
    
    def _create_statistics_view(self):
        """Create statistics visualization area"""
        from ui_components import create_stat_card
        
        stats_container = ctk.CTkFrame(self.stats_tab)
        stats_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Statistics cards
        cards_frame = ctk.CTkFrame(stats_container)
        cards_frame.pack(fill="x", pady=(0, 20))
        
        # Create stat cards
        self.stat_cards = {
            'total': create_stat_card(cards_frame, "Total Objects", "0", "📊"),
            'classes': create_stat_card(cards_frame, "Unique Classes", "0", "🏷️"),
            'confidence': create_stat_card(cards_frame, "Avg Confidence", "0%", "🎯"),
            'time': create_stat_card(cards_frame, "Processing Time", "0ms", "⏱️")
        }
        
        for card in self.stat_cards.values():
            card.pack(side="left", padx=10, expand=True, fill="x")
        
        # Class distribution area
        self.class_dist_frame = ctk.CTkFrame(stats_container)
        self.class_dist_frame.pack(fill="both", expand=True)
        
        ctk.CTkLabel(
            self.class_dist_frame,
            text="Class Distribution",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=10)
        
        self.class_dist_text = ctk.CTkTextbox(
            self.class_dist_frame,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.class_dist_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    def _create_status_bar(self):
        """Create status bar at the bottom"""
        self.status_bar = ctk.CTkFrame(self, height=30)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        
        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Ready",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="left", padx=10)
        
        self.progress_bar = ctk.CTkProgressBar(self.status_bar, width=200)
        self.progress_bar.pack(side="right", padx=10)
        self.progress_bar.set(0)
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.bind("<Control-o>", lambda e: self._load_image())
        self.bind("<Control-v>", lambda e: self._load_video())
        self.bind("<Control-w>", lambda e: self._switch_to_camera())
        self.bind("<Control-e>", lambda e: self._export_results())
        self.bind("<Control-s>", lambda e: self._take_snapshot())
        self.bind("<space>", lambda e: self._toggle_camera())
        self.bind("<Escape>", lambda e: self._stop_detection())
    
    def _update_status(self, message):
        """Update status bar message"""
        self.status_label.configure(text=message)
    
    def _load_initial_model(self):
        """Load the initial YOLO model"""
        self._update_status("Loading YOLO model...")
        self.progress_bar.set(0.5)
        
        try:
            # Get device setting
            device = self._get_device_string()
            self.detector = YOLODetector(config.DEFAULT_MODEL, device=device)
            self._update_device_info()
            self._update_status("Model loaded successfully")
            self.progress_bar.set(1)
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {e}")
            self._update_status("Model loading failed")
        finally:
            self.after(1000, lambda: self.progress_bar.set(0))
    
    def _on_model_change(self, choice):
        """Handle model selection change"""
        model_path = config.AVAILABLE_MODELS[choice]
        self._update_status(f"Loading {choice}...")
        self.progress_bar.set(0.5)
        
        try:
            self.detector.load_model(model_path)
            self._update_device_info()
            self._update_status(f"Loaded {choice}")
            self.progress_bar.set(1)
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {e}")
            self._update_status("Model loading failed")
        finally:
            self.after(1000, lambda: self.progress_bar.set(0))
    
    def _on_device_change(self, choice):
        """Handle device selection change"""
        if not self.detector:
            return
        
        self._update_status(f"Switching to {choice}...")
        device = self._get_device_string()
        actual_device = self.detector.set_device(device)
        self._update_device_info()
        
        # Reload model on new device
        if self.detector.model:
            self.detector.model.to(actual_device)
        
        self._update_status(f"Switched to {actual_device.upper()}")
    
    def _get_device_string(self):
        """Convert UI device selection to device string"""
        choice = self.device_var.get()
        if choice == "Auto":
            return "auto"
        elif choice == "CPU":
            return "cpu"
        elif choice == "GPU (CUDA)":
            return "cuda"
        return "auto"
    
    def _update_device_info(self):
        """Update device information display"""
        if not self.detector:
            return
        
        info = self.detector.get_device_info()
        device_text = f"Device: {info['device'].upper()}"
        
        if info['device'] == 'cuda':
            device_text = f"GPU: {info['device_name']}"
            if 'memory_info' in info and info['memory_info']:
                device_text += f"\nVRAM: {info['memory_info']['free']} free"
        
        self.device_info_label.configure(text=device_text)
    
    def _update_frame_skip(self, value):
        """Update frame skip setting"""
        self.frame_skip = int(value)
        if self.frame_skip == 0:
            text = "0 (Process all frames)"
        elif self.frame_skip == 1:
            text = "1 (Process every 2nd frame)"
        else:
            text = f"{self.frame_skip} (Process every {self.frame_skip+1} frames)"
        self.frame_skip_label.configure(text=text)
    
    def _update_confidence_label(self, value):
        """Update confidence threshold label"""
        self.confidence_label.configure(text=f"{value:.2f}")
    
    def _update_iou_label(self, value):
        """Update IOU threshold label"""
        self.iou_label.configure(text=f"{value:.2f}")
    
    def _load_image(self):
        """Load an image file"""
        # Stop camera if running
        if self.current_video == "webcam" and self.is_detecting:
            self._stop_detection()
        
        # Hide video controls if visible
        if hasattr(self, 'video_controls_frame'):
            self.video_controls_frame.pack_forget()
        
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image = file_path
            self.current_video = None
            self._display_image(file_path)
            self._update_status(f"Loaded: {Path(file_path).name}")
            self.source_label.configure(text="📷 Image")
            self.camera_status.configure(text="● Camera Stopped", text_color="#e53935")
            self._run_detection()  # Auto-run detection on image
    
    def _load_video(self):
        """Load a video file"""
        # Stop camera if running
        if self.current_video == "webcam" and self.is_detecting:
            self._stop_detection()
        
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_video = "video"
            self.current_video_path = file_path
            self.current_image = None
            self.video_paused = False
            self._display_video_frame(file_path)
            self._update_status(f"Loaded: {Path(file_path).name}")
            self.source_label.configure(text="🎥 Video")
            self.camera_status.configure(text="● Camera Stopped", text_color="#e53935")
            # Show video controls
            self.video_controls_frame.pack(side="bottom", fill="x", padx=10, pady=5)
            # Auto-start video processing
            self._run_detection()
    
    def _switch_to_camera(self):
        """Switch to camera input"""
        if self.current_video != "webcam":
            # Hide video controls if visible
            if hasattr(self, 'video_controls_frame'):
                self.video_controls_frame.pack_forget()
            
            self.current_video = "webcam"
            self.current_image = None
            self.source_label.configure(text="📹 Live Camera")
            self.camera_status.configure(text="● Camera Active", text_color="#43a047")
            self._update_status("Switched to camera")
            self._run_webcam_detection()
    
    def _toggle_camera(self):
        """Toggle camera pause/resume"""
        if self.current_video == "webcam":
            if self.is_detecting:
                self._stop_detection()
                self.camera_btn.configure(text="▶ Resume Camera", fg_color="green")
                self.camera_status.configure(text="● Camera Paused", text_color="#fb8c00")
            else:
                self._run_webcam_detection()
                self.camera_btn.configure(text="⏸ Pause Camera", fg_color="orange")
                self.camera_status.configure(text="● Camera Active", text_color="#43a047")
    
    def _start_default_camera(self):
        """Start camera by default on application launch"""
        if self.detector and self.camera_auto_start:
            self._update_status("Starting camera...")
            self._run_webcam_detection()
    
    def _take_snapshot(self):
        """Take a snapshot from current camera feed"""
        if self.current_video == "webcam" and hasattr(self, 'last_frame'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = config.OUTPUT_DIR / f"snapshot_{timestamp}.jpg"
            cv2.imwrite(str(snapshot_path), self.last_frame)
            self._update_status(f"Snapshot saved: {snapshot_path.name}")
            messagebox.showinfo("Snapshot Saved", f"Snapshot saved to:\n{snapshot_path}")
        else:
            messagebox.showwarning("No Camera Feed", "Camera must be active to take a snapshot")
    
    def _display_image(self, image_path):
        """Display an image in the canvas"""
        try:
            image = Image.open(image_path)
            display_size = config.MAX_IMAGE_DISPLAY_SIZE
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ctk.CTkImage(
                light_image=image,
                dark_image=image,
                size=image.size
            )
            
            self.canvas.configure(image=photo, text="")
            self.canvas.image = photo
            
        except Exception as e:
            messagebox.showerror("Display Error", f"Failed to display image: {e}")
    
    def _display_video_frame(self, video_path):
        """Display first frame of video"""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                display_size = config.MAX_IMAGE_DISPLAY_SIZE
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                photo = ctk.CTkImage(
                    light_image=image,
                    dark_image=image,
                    size=image.size
                )
                
                self.canvas.configure(image=photo, text="")
                self.canvas.image = photo
            
            cap.release()
            
        except Exception as e:
            messagebox.showerror("Display Error", f"Failed to display video: {e}")
    
    def _run_detection(self):
        """Run object detection on current input"""
        if not self.detector:
            messagebox.showwarning("No Model", "Please wait for model to load")
            return
        
        if self.current_image:
            self._detect_on_image()
        elif self.current_video == "webcam":
            if not self.is_detecting:
                self._run_webcam_detection()
        elif self.current_video == "video":
            self._detect_on_video()
        else:
            messagebox.showinfo("No Input", "Please load an image or video first")
    
    def _detect_on_image(self):
        """Run detection on current image"""
        from ui_components import update_stat_card, update_results_display
        
        self._update_status("Running detection...")
        self.progress_bar.set(0.5)
        
        confidence = self.confidence_slider.get()
        iou = self.iou_slider.get()
        
        results = self.detector.detect_image(self.current_image, confidence, iou)
        
        if results['success']:
            annotated = results['annotated_image']
            self._display_cv2_image(annotated)
            
            self.detection_results = results['detections']
            update_results_display(self.results_text, results)
            self._update_statistics(results)
            
            self._update_status(f"Detection complete: {results['total_objects']} objects found")
        else:
            messagebox.showerror("Detection Error", results['error'])
            self._update_status("Detection failed")
        
        self.progress_bar.set(0)
    
    def _detect_on_video(self):
        """Run detection on video file"""
        from ui_components import update_results_display
        
        if not self.current_video_path:
            return
            
        self._update_status("Processing video...")
        self.is_detecting = True
        self.video_paused = False
        
        # Update video controls
        self.video_play_btn.configure(text="⏸")
        
        confidence = self.confidence_slider.get()
        iou = self.iou_slider.get()
        
        def video_callback(frame, detections, current_frame, total_frames, fps):
            if self.video_paused:
                return
                
            self.video_frame_count = current_frame
            self.video_total_frames = total_frames
            
            self.after(0, lambda: self._display_cv2_image(frame))
            
            # Update progress slider if not seeking
            if not self.video_seeking:
                progress = (current_frame / total_frames * 100) if total_frames > 0 else 0
                self.after(0, lambda: self.video_slider.set(progress))
            
            # Update time display
            current_time = self._frames_to_time(current_frame, self.video_fps)
            total_time = self._frames_to_time(total_frames, self.video_fps)
            self.after(0, lambda: self.video_time_label.configure(
                text=f"{current_time} / {total_time}"
            ))
            self.after(0, lambda: self.video_info_label.configure(
                text=f"Frame {current_frame}/{total_frames} | {fps:.1f} FPS"
            ))
            self.after(0, lambda: self._update_status(
                f"Processing: Frame {current_frame}/{total_frames} | FPS: {fps:.1f}"
            ))
            self.after(0, lambda: self.fps_label.configure(text=f"FPS: {fps:.1f}"))
            
            if detections:
                self.detection_results = detections
                self.after(0, lambda: update_results_display(
                    self.results_text,
                    {'detections': detections, 'total_objects': len(detections)}
                ))
                self.after(0, lambda: self._update_statistics(
                    {'detections': detections, 'total_objects': len(detections)}
                ))
        
        # Run in thread
        def process_video():
            # Create new video capture with frame skip support
            cap = cv2.VideoCapture(self.current_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = cap.get(cv2.CAP_PROP_FPS)
            self.video_total_frames = total_frames
            frame_count = self.video_frame_count if self.video_frame_count > 0 else 0
            
            # Seek to current position if needed
            if frame_count > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            
            fps_start_time = datetime.now()
            fps_frame_count = 0
            current_fps = 0
            
            while cap.isOpened():
                if self.stop_video.is_set():
                    break
                    
                if self.video_paused:
                    time.sleep(0.1)
                    continue
                    
                # Handle seeking
                if self.video_seeking:
                    seek_frame = int(self.video_slider.get() / 100 * total_frames)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
                    frame_count = seek_frame
                    self.video_frame_count = seek_frame
                    self.video_seeking = False
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Calculate FPS
                fps_frame_count += 1
                elapsed = (datetime.now() - fps_start_time).total_seconds()
                if elapsed > 1.0:
                    current_fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_start_time = datetime.now()
                
                # Frame skipping
                if self.frame_skip > 0 and (frame_count - 1) % (self.frame_skip + 1) != 0:
                    # Show frame without detection
                    cv2.putText(
                        frame,
                        f"FPS: {current_fps:.1f} | Frame: {frame_count}/{total_frames} | Skipped",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2
                    )
                    video_callback(frame, [], frame_count, total_frames, current_fps)
                    continue
                
                # Run detection
                results = self.detector.model(
                    frame,
                    conf=confidence,
                    iou=iou,
                    verbose=False
                )
                
                # Process results
                detections = self.detector._process_results(results[0])
                annotated_frame = self.detector._draw_detections(frame, detections)
                
                # Add info overlay
                cv2.putText(
                    annotated_frame,
                    f"FPS: {current_fps:.1f} | Frame: {frame_count}/{total_frames}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                video_callback(annotated_frame, detections, frame_count, total_frames, current_fps)
                
                # Apply speed control
                if self.video_speed != 1.0:
                    delay = (1.0 / self.video_fps) / self.video_speed
                    time.sleep(max(0, delay))
            
            cap.release()
            
            # Check if should loop
            if self.video_loop_var.get() and not self.stop_video.is_set():
                self.video_frame_count = 0
                self.after(0, lambda: self._detect_on_video())
            else:
                self.is_detecting = False
                self.after(0, lambda: self.video_play_btn.configure(text="▶"))
                self.after(0, lambda: self._update_status(f"Video complete: {frame_count} frames processed"))
                self.after(0, lambda: self.fps_label.configure(text="FPS: 0.0"))
        
        self.stop_video.clear()
        self.video_thread = threading.Thread(target=process_video, daemon=True)
        self.video_thread.start()
    
    def _run_webcam_detection(self):
        """Run real-time webcam detection"""
        if self.is_detecting:
            return
        
        self.is_detecting = True
        self.camera_btn.configure(text="⏸ Pause Camera", fg_color="orange")
        self.camera_status.configure(text="● Camera Active", text_color="#43a047")
        self._update_status("Camera running...")
        
        self.stop_video.clear()
        self.video_thread = threading.Thread(target=self._process_webcam, daemon=True)
        self.video_thread.start()
    
    def _process_webcam(self):
        """Process webcam feed"""
        from ui_components import update_results_display
        
        confidence = self.confidence_slider.get()
        iou = self.iou_slider.get()
        
        def frame_callback(frame, detections, fps=0):
            # Store last frame for snapshot
            self.last_frame = frame.copy()
            self.current_fps = fps
            
            self.after(0, lambda: self._display_cv2_image(frame))
            self.after(0, lambda: self._update_status(f"Camera: {len(detections)} objects detected"))
            self.after(0, lambda: self.fps_label.configure(text=f"FPS: {fps:.1f}"))
            
            if detections:
                self.detection_results = detections
                self.after(0, lambda: update_results_display(
                    self.results_text,
                    {'detections': detections, 'total_objects': len(detections)}
                ))
                self.after(0, lambda: self._update_statistics(
                    {'detections': detections, 'total_objects': len(detections)}
                ))
        
        self.detector.detect_webcam(
            confidence, 
            iou, 
            callback=frame_callback, 
            stop_event=self.stop_video,
            frame_skip=self.frame_skip
        )
        
        self.is_detecting = False
        self.after(0, lambda: self.camera_btn.configure(text="▶ Resume Camera", fg_color="green"))
        self.after(0, lambda: self.camera_status.configure(text="● Camera Stopped", text_color="#e53935"))
        self.after(0, lambda: self.fps_label.configure(text="FPS: 0.0"))
    
    def _stop_detection(self):
        """Stop ongoing detection"""
        self.stop_video.set()
        self.is_detecting = False
        if self.current_video == "webcam":
            self.camera_btn.configure(text="▶ Resume Camera", fg_color="green")
            self.camera_status.configure(text="● Camera Paused", text_color="#fb8c00")
        elif self.current_video == "video":
            self.video_paused = True
            if hasattr(self, 'video_play_btn'):
                self.video_play_btn.configure(text="▶ Play")
        self._update_status("Detection stopped")
        self.progress_bar.set(0)
    
    def _toggle_video_playback(self):
        """Toggle video play/pause"""
        if self.current_video == "video":
            self.video_paused = not self.video_paused
            if self.video_paused:
                self.video_play_btn.configure(text="▶")
                self._update_status("Video paused")
            else:
                self.video_play_btn.configure(text="⏸")
                self._update_status("Video playing")
                if not self.is_detecting:
                    self._detect_on_video()
    
    def _stop_video_playback(self):
        """Stop video playback completely"""
        self.stop_video.set()
        self.video_paused = True
        self.is_detecting = False
        self.video_frame_count = 0
        self.video_slider.set(0)
        self.video_time_label.configure(text="0:00 / 0:00")
        self.video_play_btn.configure(text="▶")
        self._update_status("Video stopped")
    
    def _restart_video(self):
        """Restart video from beginning"""
        self.video_frame_count = 0
        self.video_slider.set(0)
        self.video_paused = False
        if not self.is_detecting:
            self._detect_on_video()
        self._update_status("Video restarted")
    
    def _on_video_seek(self, value):
        """Handle video seek slider change"""
        if self.current_video == "video" and self.video_total_frames > 0:
            self.video_seeking = True
            seek_frame = int(value / 100 * self.video_total_frames)
            self.video_frame_count = seek_frame
            # Update time display immediately
            current_time = self._frames_to_time(seek_frame, self.video_fps)
            total_time = self._frames_to_time(self.video_total_frames, self.video_fps)
            self.video_time_label.configure(text=f"{current_time} / {total_time}")
    
    def _on_speed_change(self, value):
        """Handle video speed change"""
        speed_map = {
            "0.25x": 0.25,
            "0.5x": 0.5,
            "1.0x": 1.0,
            "1.5x": 1.5,
            "2.0x": 2.0
        }
        self.video_speed = speed_map.get(value, 1.0)
        self._update_status(f"Playback speed: {value}")
    
    def _frames_to_time(self, frames, fps):
        """Convert frame number to time string"""
        if fps <= 0:
            return "0:00"
        seconds = int(frames / fps)
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}:{seconds:02d}"
    
    def _display_cv2_image(self, cv_image):
        """Display OpenCV image in canvas"""
        try:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_image)
            
            display_size = config.MAX_IMAGE_DISPLAY_SIZE
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
            
            self.canvas.configure(image=photo, text="")
            self.canvas.image = photo
            
        except Exception as e:
            print(f"Display error: {e}")
    
    def _update_statistics(self, results):
        """Update statistics display"""
        from ui_components import update_stat_card
        
        if not results.get('detections'):
            return
        
        stats = self.detector.get_statistics(results['detections'])
        
        update_stat_card(self.stat_cards['total'], str(stats['total_objects']))
        update_stat_card(self.stat_cards['classes'], str(stats['unique_classes']))
        update_stat_card(self.stat_cards['confidence'], f"{stats['average_confidence']:.1%}")
        
        if 'processing_time' in results:
            update_stat_card(self.stat_cards['time'], f"{results['processing_time']:.1f}ms")
        
        self.class_dist_text.delete("1.0", "end")
        dist_text = "Class Distribution:\n" + "=" * 40 + "\n"
        
        for class_name, count in sorted(stats['class_distribution'].items(), key=lambda x: x[1], reverse=True):
            bar_length = int((count / stats['total_objects']) * 30)
            bar = "█" * bar_length
            dist_text += f"{class_name:20} {bar} {count}\n"
        
        self.class_dist_text.insert("1.0", dist_text)
    
    def _export_results(self):
        """Export detection results"""
        if not self.detection_results:
            messagebox.showinfo("No Results", "No detection results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=f".{self.export_format.get().lower()}",
            filetypes=[
                (f"{self.export_format.get()} files", f"*.{self.export_format.get().lower()}"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                output_file = self.detector.export_results(
                    self.detection_results,
                    self.export_format.get(),
                    file_path.rsplit('.', 1)[0]
                )
                messagebox.showinfo("Export Success", f"Results exported to {output_file}")
                self._update_status(f"Exported to {Path(output_file).name}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")


    def _change_model_from_menu(self, model_name):
        """Change model from menu selection"""
        self.model_var.set(model_name)
        self._on_model_change(model_name)
    
    def _show_about(self):
        """Show about dialog"""
        about_text = f"""
🚀 {config.APP_NAME}
Version {config.APP_VERSION}

Advanced Object Detection System
Powered by YOLOv8 (Ultralytics)

Core Features:
• Real-time camera detection
• Multi-source processing (Camera/Image/Video)
• 5 YOLO models (Nano to Extra Large)
• Multiple export formats
• Snapshot capability
• Live statistics dashboard

Built with CustomTkinter & OpenCV
        """
        messagebox.showinfo("About YOLO Krypton", about_text)
    
    def _show_shortcuts(self):
        """Show keyboard shortcuts"""
        shortcuts_text = """
Keyboard Shortcuts:

Ctrl+O    - Open Image
Ctrl+V    - Open Video
Ctrl+W    - Switch to Camera
Ctrl+E    - Export Results
Ctrl+S    - Take Snapshot
Space     - Pause/Resume Camera
Esc       - Stop Detection
        """
        messagebox.showinfo("Keyboard Shortcuts", shortcuts_text)


if __name__ == "__main__":
    app = YOLODetectionApp()
    app.mainloop()
