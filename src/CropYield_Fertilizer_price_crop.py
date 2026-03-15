import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import seaborn as sns
import threading
import time
import random
from PIL import Image, ImageTk
import io
import base64
import requests
import traceback
import os

# Define the main application class
class SoilAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Soil Analysis & Micronutrient Classification for IoT Farms")
        self.root.geometry("1400x800")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize variables
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = None
        self.Y = None
        self.le = None
        self.model = None
        self.ms = MinMaxScaler()
        self.sc = StandardScaler()
        self.rfc = RandomForestClassifier(random_state=42)
        self.iot_simulation_running = False
        self.iot_thread = None
        
        # Initialize dictionaries and data structures
        self.natural_fertilizers = {
            "rice": ["Compost", "Manure", "NPK Fertilizer"],
            "maize": ["Compost", "Manure", "NPK Fertilizer"],
            "jute": ["Organic Fertilizer", "NPK Fertilizer"],
            "cotton": ["Compost", "Manure", "NPK Fertilizer"],
            "coconut": ["Organic Fertilizer", "Palm Ash", "Fish Emulsion"],
            "papaya": ["Compost", "Manure", "Organic Fertilizer"],
            "orange": ["Compost", "Manure", "Citrus Fertilizer"],
            "apple": ["Compost", "Manure", "Fruit Tree Fertilizer"],
            "muskmelon": ["Compost", "Manure", "Melon Fertilizer"],
            "watermelon": ["Compost", "Manure", "Melon Fertilizer"],
            "grapes": ["Compost", "Manure", "Grape Vine Fertilizer"],
            "mango": ["Compost", "Manure", "Fruit Tree Fertilizer"],
            "banana": ["Compost", "Manure", "Banana Fertilizer"],
            "pomegranate": ["Compost", "Manure", "Fruit Tree Fertilizer"],
            "lentil": ["Compost", "Manure", "Legume Fertilizer"],
            "blackgram": ["Compost", "Manure", "Legume Fertilizer"],
            "mungbean": ["Compost", "Manure", "Legume Fertilizer"],
            "mothbeans": ["Compost", "Manure", "Legume Fertilizer"],
            "pigeonpeas": ["Compost", "Manure", "Legume Fertilizer"],
            "kidneybeans": ["Compost", "Manure", "Legume Fertilizer"],
            "chickpea": ["Compost", "Manure", "Legume Fertilizer"],
            "coffee": ["Compost", "Manure", "Coffee Plant Fertilizer"],
        }
        
        self.price = {
            "rice": ["23,450 per Acre"],
            "maize": ["19,450 per Acre"],
            "jute": ["33,450 per Acre"],
            "cotton": ["19,250 per Acre"],
            "coconut": ["43,450 per Acre"],
            "papaya": ["12,470 per Acre"],
            "orange": ["53,450 per Acre"],
            "apple": ["29,450 per Acre"],
            "muskmelon": ["63,450 per Acre"],
            "watermelon": ["83,450 per Acre"],
            "grapes": ["73,450 per Acre"],
            "mango": ["73,450 per Acre"],
            "banana": ["83,450 per Acre"],
            "pomegranate": ["63,450 per Acre"],
            "lentil": ["93,450 per Acre"],
            "blackgram": ["53,450 per Acre"],
            "mungbean": ["13,450 per Acre"],
            "mothbeans": ["43,450 per Acre"],
            "pigeonpeas": ["23,450 per Acre"],
            "kidneybeans": ["33,450 per Acre"],
            "chickpea": ["33,450 per Acre"],
            "coffee": ["23,450 per Acre"],
        }
        
        # Micronutrient thresholds
        self.micronutrient_thresholds = {
            'Iron': {'low': 2.5, 'optimal': 4.5, 'high': 6.0},
            'Zinc': {'low': 0.5, 'optimal': 1.0, 'high': 2.0},
            'Manganese': {'low': 1.0, 'optimal': 2.0, 'high': 5.0},
            'Copper': {'low': 0.2, 'optimal': 0.5, 'high': 1.0},
            'Boron': {'low': 0.5, 'optimal': 1.0, 'high': 2.0},
            'Molybdenum': {'low': 0.01, 'optimal': 0.05, 'high': 0.1}
        }
        
        # Soil types and their characteristics
        self.soil_types = {
            'Clay': {
                'description': 'Heavy soil with small particles, retains water and nutrients well',
                'suitable_crops': ['rice', 'wheat', 'cotton'],
                'drainage': 'Poor',
                'water_retention': 'High'
            },
            'Sandy': {
                'description': 'Light soil with large particles, drains quickly but retains few nutrients',
                'suitable_crops': ['carrots', 'potatoes', 'lettuce'],
                'drainage': 'Excellent',
                'water_retention': 'Low'
            },
            'Loam': {
                'description': 'Balanced soil with good structure, ideal for most crops',
                'suitable_crops': ['corn', 'soybeans', 'vegetables'],
                'drainage': 'Good',
                'water_retention': 'Medium'
            },
            'Silt': {
                'description': 'Smooth soil with medium particles, holds moisture well',
                'suitable_crops': ['fruits', 'vegetables', 'grasses'],
                'drainage': 'Moderate',
                'water_retention': 'Medium-High'
            },
            'Peat': {
                'description': 'Organic soil with high acidity, good for acid-loving plants',
                'suitable_crops': ['blueberries', 'cranberries', 'rhododendrons'],
                'drainage': 'Variable',
                'water_retention': 'High'
            },
            'Volcanic': {
                'description': 'Rich in minerals from volcanic activity, very fertile',
                'suitable_crops': ['coffee', 'tea', 'vegetables'],
                'drainage': 'Good',
                'water_retention': 'Medium'
            }
        }
        
        # Initialize IoT simulation data
        self.iot_data = {
            'temperature': [],
            'humidity': [],
            'soil_moisture': [],
            'nitrogen': [],
            'phosphorus': [],
            'potassium': []
        }
        
        # Create styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', font=('Arial', 12), background='#4CAF50', foreground='white')
        self.style.configure('TLabel', font=('Arial', 12), background='#f0f0f0')
        self.style.configure('Header.TLabel', font=('Arial', 16, 'bold'), background='#2E7D32', foreground='white')
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TSeparator', background='#bdbdbd')
        
        # Create main frames
        self.create_header_frame()
        self.create_main_content()
        
        # Initialize IoT data with some default values to prevent empty data errors
        self.initialize_iot_data()
        
        # Set up a protocol for when the window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def on_closing(self):
        """Handle window closing event"""
        # Stop IoT simulation if running
        if self.iot_simulation_running:
            self.iot_simulation_running = False
            if self.iot_thread and self.iot_thread.is_alive():
                self.iot_thread.join(timeout=1.0)
        
        # Destroy the window
        self.root.destroy()
        
    def initialize_iot_data(self):
        """Initialize IoT data with some default values"""
        # Add some initial data points to prevent empty data errors
        for _ in range(5):
            self.iot_data['temperature'].append(random.uniform(20.0, 35.0))
            self.iot_data['humidity'].append(random.uniform(30.0, 90.0))
            self.iot_data['soil_moisture'].append(random.uniform(20.0, 80.0))
            self.iot_data['nitrogen'].append(random.uniform(30.0, 150.0))
            self.iot_data['phosphorus'].append(random.uniform(20.0, 100.0))
            self.iot_data['potassium'].append(random.uniform(20.0, 120.0))
        
    def create_header_frame(self):
        """Create the header frame with title and logo"""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(header_frame, 
                               text="Soil Analysis & Micronutrient Classification for IoT Farms",
                               font=('Arial', 20, 'bold'),
                               foreground='#2E7D32',
                               background='#f0f0f0')
        title_label.pack(side=tk.LEFT, padx=20)
        
        # Status indicator
        self.status_frame = ttk.Frame(header_frame, width=20, height=20)
        self.status_frame.pack(side=tk.RIGHT, padx=20)
        
        self.status_indicator = tk.Canvas(self.status_frame, width=20, height=20, bg='#f0f0f0', highlightthickness=0)
        self.status_indicator.create_oval(2, 2, 18, 18, fill='gray', outline='')
        self.status_indicator.pack()
        
        self.status_label = ttk.Label(self.status_frame, text="IoT Status: Disconnected", background='#f0f0f0')
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
    def create_main_content(self):
        """Create the main content area with tabs"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.analysis_tab = ttk.Frame(self.notebook)
        self.prediction_tab = ttk.Frame(self.notebook)
        self.iot_tab = ttk.Frame(self.notebook)
        self.micronutrient_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.data_tab, text="Data Management")
        self.notebook.add(self.analysis_tab, text="Soil Analysis")
        self.notebook.add(self.prediction_tab, text="Crop Prediction")
        self.notebook.add(self.iot_tab, text="IoT Monitoring")
        self.notebook.add(self.micronutrient_tab, text="Micronutrient Analysis")
        
        # Setup each tab
        self.setup_data_tab()
        self.setup_analysis_tab()
        self.setup_prediction_tab()
        self.setup_iot_tab()
        self.setup_micronutrient_tab()
        
    def setup_data_tab(self):
        """Setup the data management tab"""
        # Left frame for controls
        left_frame = ttk.Frame(self.data_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Upload button
        upload_btn = ttk.Button(left_frame, text="Upload Dataset", command=self.upload_dataset)
        upload_btn.pack(fill=tk.X, pady=5)
        
        # Load from URL button
        url_btn = ttk.Button(left_frame, text="Load from URL", command=self.load_from_url)
        url_btn.pack(fill=tk.X, pady=5)
        
        # Process button
        process_btn = ttk.Button(left_frame, text="Process Dataset", command=self.process_dataset)
        process_btn.pack(fill=tk.X, pady=5)
        
        # Export button
        export_btn = ttk.Button(left_frame, text="Export Results", command=self.export_results)
        export_btn.pack(fill=tk.X, pady=5)
        
        # Dataset info frame
        info_frame = ttk.LabelFrame(left_frame, text="Dataset Information")
        info_frame.pack(fill=tk.X, pady=10, padx=5)
        
        self.dataset_info = tk.Text(info_frame, height=10, width=30, wrap=tk.WORD)
        self.dataset_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right frame for data display
        right_frame = ttk.Frame(self.data_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Data display
        self.data_display_frame = ttk.LabelFrame(right_frame, text="Dataset Preview")
        self.data_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create Treeview for data display
        self.data_tree = ttk.Treeview(self.data_display_frame)
        self.data_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Add scrollbar to treeview
        scrollbar = ttk.Scrollbar(self.data_display_frame, orient="vertical", command=self.data_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        
        # Path label
        self.path_label = ttk.Label(right_frame, text="No dataset loaded", foreground='gray')
        self.path_label.pack(anchor=tk.W, padx=5)
        
    def setup_analysis_tab(self):
        """Setup the soil analysis tab"""
        # Left frame for controls and info
        left_frame = ttk.Frame(self.analysis_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10, expand=False)
        
        # Analysis controls
        controls_frame = ttk.LabelFrame(left_frame, text="Analysis Controls")
        controls_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Train model button
        train_btn = ttk.Button(controls_frame, text="Train Model", command=self.train_model)
        train_btn.pack(fill=tk.X, pady=5, padx=5)
        
        # Evaluate model button
        evaluate_btn = ttk.Button(controls_frame, text="Evaluate Model", command=self.evaluate_model)
        evaluate_btn.pack(fill=tk.X, pady=5, padx=5)
        
        # Soil type analysis button
        soil_analysis_btn = ttk.Button(controls_frame, text="Analyze Soil Types", command=self.analyze_soil_types)
        soil_analysis_btn.pack(fill=tk.X, pady=5, padx=5)
        
        # Results text area
        results_frame = ttk.LabelFrame(left_frame, text="Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)
        
        self.analysis_results = tk.Text(results_frame, wrap=tk.WORD)
        self.analysis_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right frame for visualizations
        right_frame = ttk.Frame(self.analysis_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(right_frame, text="Visualizations")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a frame for the matplotlib figure
        self.fig_frame = ttk.Frame(viz_frame)
        self.fig_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize with empty figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_prediction_tab(self):
        """Setup the crop prediction tab"""
        # Left frame for input parameters
        left_frame = ttk.Frame(self.prediction_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Input parameters frame
        input_frame = ttk.LabelFrame(left_frame, text="Input Parameters")
        input_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Create input fields
        self.input_vars = {}
        parameters = [
            ("N (Nitrogen)", "N"),
            ("P (Phosphorus)", "P"),
            ("K (Potassium)", "K"),
            ("Temperature (°C)", "temperature"),
            ("Humidity (%)", "humidity"),
            ("pH", "pH"),
            ("Rainfall (mm)", "rainfall")
        ]
        
        for i, (label_text, var_name) in enumerate(parameters):
            frame = ttk.Frame(input_frame)
            frame.pack(fill=tk.X, pady=2)
            
            label = ttk.Label(frame, text=label_text, width=15)
            label.pack(side=tk.LEFT, padx=5)
            
            self.input_vars[var_name] = tk.StringVar()
            entry = ttk.Entry(frame, textvariable=self.input_vars[var_name], width=15)
            entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Add buttons
        buttons_frame = ttk.Frame(left_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        predict_btn = ttk.Button(buttons_frame, text="Predict Crop", command=self.predict_crop)
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(buttons_frame, text="Clear Fields", command=self.clear_prediction_fields)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Use IoT data button
        iot_data_btn = ttk.Button(buttons_frame, text="Use IoT Data", command=self.use_iot_data)
        iot_data_btn.pack(side=tk.LEFT, padx=5)
        
        # Right frame for prediction results
        right_frame = ttk.Frame(self.prediction_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(right_frame, text="Prediction Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a frame for the results display
        self.prediction_results_frame = ttk.Frame(results_frame)
        self.prediction_results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize with empty labels
        self.crop_result_label = ttk.Label(self.prediction_results_frame, 
                                          text="No prediction yet", 
                                          font=('Arial', 16, 'bold'),
                                          foreground='#2E7D32')
        self.crop_result_label.pack(pady=10)
        
        # Fertilizer recommendations
        self.fertilizer_frame = ttk.LabelFrame(self.prediction_results_frame, text="Recommended Fertilizers")
        self.fertilizer_frame.pack(fill=tk.X, pady=5)
        
        self.fertilizer_label = ttk.Label(self.fertilizer_frame, text="N/A")
        self.fertilizer_label.pack(pady=5, padx=5)
        
        # Price estimation
        self.price_frame = ttk.LabelFrame(self.prediction_results_frame, text="Estimated Price")
        self.price_frame.pack(fill=tk.X, pady=5)
        
        self.price_label = ttk.Label(self.price_frame, text="N/A")
        self.price_label.pack(pady=5, padx=5)
        
        # Suitability score
        self.suitability_frame = ttk.LabelFrame(self.prediction_results_frame, text="Suitability Score")
        self.suitability_frame.pack(fill=tk.X, pady=5)
        
        self.suitability_canvas = tk.Canvas(self.suitability_frame, height=30, bg='white')
        self.suitability_canvas.pack(fill=tk.X, pady=5, padx=5)
        
        # Prediction history
        history_frame = ttk.LabelFrame(right_frame, text="Prediction History")
        history_frame.pack(fill=tk.X, pady=10, padx=5)
        
        self.history_text = tk.Text(history_frame, height=6, wrap=tk.WORD)
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def setup_iot_tab(self):
        """Setup the IoT monitoring tab"""
        # Control panel on the left
        control_panel = ttk.Frame(self.iot_tab)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # IoT controls
        controls_frame = ttk.LabelFrame(control_panel, text="IoT Controls")
        controls_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Start/Stop IoT simulation
        self.iot_btn_text = tk.StringVar(value="Start IoT Simulation")
        iot_btn = ttk.Button(controls_frame, textvariable=self.iot_btn_text, command=self.toggle_iot_simulation)
        iot_btn.pack(fill=tk.X, pady=5, padx=5)
        
        # Export IoT data
        export_iot_btn = ttk.Button(controls_frame, text="Export IoT Data", command=self.export_iot_data)
        export_iot_btn.pack(fill=tk.X, pady=5, padx=5)
        
        # IoT device settings
        settings_frame = ttk.LabelFrame(control_panel, text="Device Settings")
        settings_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Sample rate
        sample_frame = ttk.Frame(settings_frame)
        sample_frame.pack(fill=tk.X, pady=5)
        
        sample_label = ttk.Label(sample_frame, text="Sample Rate (s):")
        sample_label.pack(side=tk.LEFT, padx=5)
        
        self.sample_rate = tk.StringVar(value="5")
        sample_entry = ttk.Entry(sample_frame, textvariable=self.sample_rate, width=10)
        sample_entry.pack(side=tk.LEFT, padx=5)
        
        # Device list
        devices_frame = ttk.LabelFrame(control_panel, text="Connected Devices")
        devices_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)
        
        # Simulated device list
        self.device_list = tk.Listbox(devices_frame)
        self.device_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.device_list.insert(tk.END, "Soil Sensor Node 1")
        self.device_list.insert(tk.END, "Weather Station 1")
        self.device_list.insert(tk.END, "Irrigation Controller 1")
        
        # Right side with visualizations
        viz_panel = ttk.Frame(self.iot_tab)
        viz_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Current readings
        readings_frame = ttk.LabelFrame(viz_panel, text="Current Readings")
        readings_frame.pack(fill=tk.X, pady=5)
        
        # Create a grid of current readings
        self.reading_labels = {}
        readings = [
            ("Temperature", "°C", "temperature"),
            ("Humidity", "%", "humidity"),
            ("Soil Moisture", "%", "soil_moisture"),
            ("Nitrogen", "mg/kg", "nitrogen"),
            ("Phosphorus", "mg/kg", "phosphorus"),
            ("Potassium", "mg/kg", "potassium")
        ]
        
        for i, (label_text, unit, var_name) in enumerate(readings):
            row, col = i // 3, i % 3
            frame = ttk.Frame(readings_frame)
            frame.grid(row=row, column=col, padx=10, pady=5, sticky="nsew")
            
            name_label = ttk.Label(frame, text=f"{label_text} ({unit})")
            name_label.pack(anchor=tk.W)
            
            self.reading_labels[var_name] = ttk.Label(frame, text="--", font=('Arial', 14, 'bold'))
            self.reading_labels[var_name].pack(anchor=tk.W)
        
        # Make columns expandable
        for i in range(3):
            readings_frame.columnconfigure(i, weight=1)
        
        # Historical data visualization
        history_frame = ttk.LabelFrame(viz_panel, text="Historical Data")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create a frame for the matplotlib figure
        self.iot_fig_frame = ttk.Frame(history_frame)
        self.iot_fig_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize with empty figure
        self.iot_fig, self.iot_ax = plt.subplots(figsize=(8, 4))
        self.iot_canvas = FigureCanvasTkAgg(self.iot_fig, master=self.iot_fig_frame)
        self.iot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Parameter selection for graph
        param_frame = ttk.Frame(viz_panel)
        param_frame.pack(fill=tk.X, pady=5)
        
        param_label = ttk.Label(param_frame, text="Select Parameter:")
        param_label.pack(side=tk.LEFT, padx=5)
        
        self.selected_param = tk.StringVar(value="temperature")
        param_options = ttk.Combobox(param_frame, textvariable=self.selected_param)
        param_options['values'] = ('temperature', 'humidity', 'soil_moisture', 'nitrogen', 'phosphorus', 'potassium')
        param_options.pack(side=tk.LEFT, padx=5)
        param_options.bind('<<ComboboxSelected>>', self.update_iot_graph)
        
    def setup_micronutrient_tab(self):
        """Setup the micronutrient analysis tab"""
        # Left panel for controls and input
        left_panel = ttk.Frame(self.micronutrient_tab)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Micronutrient input
        input_frame = ttk.LabelFrame(left_panel, text="Micronutrient Levels")
        input_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Create input fields for micronutrients
        self.micronutrient_vars = {}
        micronutrients = [
            ("Iron (Fe)", "Iron", "mg/kg"),
            ("Zinc (Zn)", "Zinc", "mg/kg"),
            ("Manganese (Mn)", "Manganese", "mg/kg"),
            ("Copper (Cu)", "Copper", "mg/kg"),
            ("Boron (B)", "Boron", "mg/kg"),
            ("Molybdenum (Mo)", "Molybdenum", "mg/kg")
        ]
        
        for i, (label_text, var_name, unit) in enumerate(micronutrients):
            frame = ttk.Frame(input_frame)
            frame.pack(fill=tk.X, pady=2)
            
            label = ttk.Label(frame, text=f"{label_text} ({unit}):", width=20)
            label.pack(side=tk.LEFT, padx=5)
            
            self.micronutrient_vars[var_name] = tk.StringVar()
            entry = ttk.Entry(frame, textvariable=self.micronutrient_vars[var_name], width=10)
            entry.pack(side=tk.LEFT, padx=5)
        
        # Buttons
        buttons_frame = ttk.Frame(left_panel)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        analyze_btn = ttk.Button(buttons_frame, text="Analyze Micronutrients", command=self.analyze_micronutrients)
        analyze_btn.pack(side=tk.LEFT, padx=5)
        
        random_btn = ttk.Button(buttons_frame, text="Generate Random Data", command=self.generate_random_micronutrients)
        random_btn.pack(side=tk.LEFT, padx=5)
        
        # Soil type selection
        soil_frame = ttk.LabelFrame(left_panel, text="Soil Type")
        soil_frame.pack(fill=tk.X, pady=10, padx=5)
        
        self.soil_type = tk.StringVar(value="Clay")
        soil_options = ttk.Combobox(soil_frame, textvariable=self.soil_type)
        soil_options['values'] = tuple(self.soil_types.keys())
        soil_options.pack(fill=tk.X, padx=5, pady=5)
        
        # Soil info display
        soil_info_frame = ttk.LabelFrame(left_panel, text="Soil Information")
        soil_info_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)
        
        self.soil_info_text = tk.Text(soil_info_frame, wrap=tk.WORD, height=8)
        self.soil_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel for visualizations
        right_panel = ttk.Frame(self.micronutrient_tab)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Micronutrient visualization
        viz_frame = ttk.LabelFrame(right_panel, text="Micronutrient Analysis")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a frame for the matplotlib figure
        self.micro_fig_frame = ttk.Frame(viz_frame)
        self.micro_fig_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize with empty figure
        self.micro_fig, self.micro_ax = plt.subplots(figsize=(8, 6))
        self.micro_canvas = FigureCanvasTkAgg(self.micro_fig, master=self.micro_fig_frame)
        self.micro_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Recommendations frame
        rec_frame = ttk.LabelFrame(right_panel, text="Recommendations")
        rec_frame.pack(fill=tk.X, pady=10, padx=5)
        
        self.rec_text = tk.Text(rec_frame, wrap=tk.WORD, height=6)
        self.rec_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def load_from_url(self):
        """Load dataset from a URL"""
        try:
            url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Crop_recommendation1-YP3Azvu00IRogdnMF8xpNwJkrY2aPO.csv"
            
            # Download the CSV file
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    # Try to decode the content with the current encoding
                    content = response.content.decode(encoding)
                    
                    # Create a StringIO object to simulate a file
                    from io import StringIO
                    csv_data = StringIO(content)
                    
                    # Read the CSV into a DataFrame
                    self.dataset = pd.read_csv(csv_data)
                    
                    # Clean up the Soil column (remove special characters)
                    if 'Soil' in self.dataset.columns:
                        self.dataset['Soil'] = self.dataset['Soil'].str.replace(r'[^\x00-\x7F]+', '', regex=True)
                    
                    # Convert string columns to numeric
                    for col in ['N', 'P', 'K']:
                        if col in self.dataset.columns:
                            try:
                                self.dataset[col] = pd.to_numeric(self.dataset[col], errors='coerce')
                            except Exception as e:
                                messagebox.showwarning("Warning", f"Error converting {col} to numeric: {str(e)}")
                    
                    # Update UI
                    self.path_label.config(text=f"Dataset: {url} (Encoding: {encoding})")
                    
                    # Update dataset info
                    self.dataset_info.delete(1.0, tk.END)
                    self.dataset_info.insert(tk.END, f"Rows: {self.dataset.shape[0]}\n")
                    self.dataset_info.insert(tk.END, f"Columns: {self.dataset.shape[1]}\n")
                    self.dataset_info.insert(tk.END, f"Features: {', '.join(self.dataset.columns)}\n")
                    
                    # Display data types
                    self.dataset_info.insert(tk.END, "\nData Types:\n")
                    for col, dtype in self.dataset.dtypes.items():
                        self.dataset_info.insert(tk.END, f"{col}: {dtype}\n")
                    
                    # Update treeview
                    self.update_data_treeview()
                    
                    messagebox.showinfo("Success", f"Dataset loaded successfully with {encoding} encoding!")
                    return
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to process dataset: {str(e)}")
                    return
            
            messagebox.showerror("Error", "Failed to load dataset with any common encoding. The file might be corrupted.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to download dataset: {str(e)}")
    
    def upload_dataset(self):
        """Upload and load dataset with proper encoding handling"""
        filename = filedialog.askopenfilename(initialdir=".", title="Select Dataset",
                                             filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if filename:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    self.dataset = pd.read_csv(filename, encoding=encoding)
                    
                    # Clean up the Soil column (remove special characters)
                    if 'Soil' in self.dataset.columns:
                        self.dataset['Soil'] = self.dataset['Soil'].str.replace(r'[^\x00-\x7F]+', '', regex=True)
                    
                    # Convert string columns to numeric
                    for col in ['N', 'P', 'K']:
                        if col in self.dataset.columns:
                            try:
                                self.dataset[col] = pd.to_numeric(self.dataset[col], errors='coerce')
                            except Exception as e:
                                messagebox.showwarning("Warning", f"Error converting {col} to numeric: {str(e)}")
                    
                    self.path_label.config(text=f"Dataset: {filename} (Encoding: {encoding})")
                    
                    # Update dataset info
                    self.dataset_info.delete(1.0, tk.END)
                    self.dataset_info.insert(tk.END, f"Rows: {self.dataset.shape[0]}\n")
                    self.dataset_info.insert(tk.END, f"Columns: {self.dataset.shape[1]}\n")
                    self.dataset_info.insert(tk.END, f"Features: {', '.join(self.dataset.columns)}\n")
                    
                    # Display data types
                    self.dataset_info.insert(tk.END, "\nData Types:\n")
                    for col, dtype in self.dataset.dtypes.items():
                        self.dataset_info.insert(tk.END, f"{col}: {dtype}\n")
                    
                    # Update treeview
                    self.update_data_treeview()
                    
                    messagebox.showinfo("Success", f"Dataset loaded successfully with {encoding} encoding!")
                    return
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
                    return
            
            messagebox.showerror("Error", "Failed to load dataset with any common encoding. The file might be corrupted.")
    
    def update_data_treeview(self):
        """Update the treeview with dataset"""
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Configure columns
        self.data_tree['columns'] = list(self.dataset.columns)
        self.data_tree['show'] = 'headings'
        
        for col in self.dataset.columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
        
        # Add data rows
        for i, row in self.dataset.head(50).iterrows():
            values = []
            for val in row:
                if pd.isna(val):
                    values.append("")
                else:
                    values.append(str(val))
            self.data_tree.insert('', tk.END, values=values)
    
    def process_dataset(self):
        """Process the dataset for analysis"""
        if self.dataset is None:
            messagebox.showerror("Error", "Please upload a dataset first!")
            return
        
        try:
            self.le = LabelEncoder()
            
            # Check if required columns exist
            required_columns = ['label', 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            missing_columns = [col for col in required_columns if col.lower() not in [c.lower() for c in self.dataset.columns]]
            
            if missing_columns:
                messagebox.showerror("Error", f"Missing required columns: {', '.join(missing_columns)}")
                return
            
            # Create a copy of the dataset to avoid modifying the original
            processed_df = self.dataset.copy()
            
            # Standardize column names (case insensitive matching)
            column_mapping = {}
            for req_col in required_columns:
                for col in processed_df.columns:
                    if col.lower() == req_col.lower():
                        column_mapping[col] = req_col
            
            # Rename columns to standardized names
            processed_df = processed_df.rename(columns=column_mapping)
            
            # Ensure numeric columns are properly converted
            numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            for col in numeric_cols:
                if col in processed_df.columns:
                    try:
                        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                    except Exception as e:
                        messagebox.showwarning("Warning", f"Error converting {col} to numeric: {str(e)}")
            
            # Drop rows with NaN values
            original_count = len(processed_df)
            processed_df = processed_df.dropna()
            dropped_count = original_count - len(processed_df)
            
            if dropped_count > 0:
                messagebox.showinfo("Info", f"Dropped {dropped_count} rows with missing values.")
            
            # Extract features and target
            self.X = processed_df.drop(['label'], axis=1)
            if 'Soil' in processed_df.columns:
                self.X = self.X.drop(['Soil'], axis=1)
            self.Y = processed_df['label']
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.Y, test_size=0.2, random_state=42)
            
            # Scale features
            self.X_train_scaled = self.ms.fit_transform(self.X_train)
            self.X_train_scaled = self.sc.fit_transform(self.X_train_scaled)
            self.X_test_scaled = self.ms.transform(self.X_test)
            self.X_test_scaled = self.sc.transform(self.X_test_scaled)
            
            # Train random forest classifier
            self.rfc.fit(self.X_train_scaled, self.y_train)
            
            # Update info
            self.analysis_results.delete(1.0, tk.END)
            self.analysis_results.insert(tk.END, f"Dataset processed successfully!\n\n")
            self.analysis_results.insert(tk.END, f"Total records: {len(self.X)}\n")
            self.analysis_results.insert(tk.END, f"Training samples: {len(self.X_train)}\n")
            self.analysis_results.insert(tk.END, f"Testing samples: {len(self.X_test)}\n")
            
            # Create a feature importance plot
            self.plot_feature_importance()
            
            messagebox.showinfo("Success", "Dataset processed successfully!")
        except Exception as e:
            error_msg = f"Failed to process dataset: {str(e)}\n\n{traceback.format_exc()}"
            messagebox.showerror("Error", error_msg)
            self.analysis_results.delete(1.0, tk.END)
            self.analysis_results.insert(tk.END, error_msg)
    
    def plot_feature_importance(self):
        """Plot feature importance from Random Forest"""
        if not hasattr(self, 'rfc') or not hasattr(self.rfc, 'feature_importances_'):
            return
        
        try:
            # Clear previous plot
            self.ax.clear()
            
            # Get feature importance
            importances = self.rfc.feature_importances_
            feature_names = self.X.columns
            
            # Sort by importance
            indices = np.argsort(importances)[::-1]
            
            # Plot
            self.ax.bar(range(len(importances)), importances[indices])
            self.ax.set_title('Feature Importance')
            self.ax.set_xticks(range(len(importances)))
            self.ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
            self.ax.set_xlabel('Features')
            self.ax.set_ylabel('Importance')
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot feature importance: {str(e)}")
    
    def train_model(self):
        """Train the decision tree model"""
        if not hasattr(self, 'X') or self.X is None:
            messagebox.showerror("Error", "Please process the dataset first!")
            return
        
        try:
            # Train decision tree regressor
            self.model = DecisionTreeRegressor(max_depth=100, random_state=0, 
                                              max_leaf_nodes=20, max_features=None, 
                                              splitter="random")
            self.model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            predictions = self.model.predict(self.X_test_scaled)
            mse = mean_squared_error(self.y_test, predictions)
            rmse = np.sqrt(mse)
            
            # Update results
            self.analysis_results.delete(1.0, tk.END)
            self.analysis_results.insert(tk.END, "Model Training Complete\n\n")
            self.analysis_results.insert(tk.END, f"Decision Tree RMSE: {rmse:.4f}\n")
            
            # Plot actual vs predicted
            self.plot_actual_vs_predicted(self.y_test, predictions)
            
            messagebox.showinfo("Success", "Model trained successfully!")
        except Exception as e:
            error_msg = f"Failed to train model: {str(e)}\n\n{traceback.format_exc()}"
            messagebox.showerror("Error", error_msg)
            self.analysis_results.delete(1.0, tk.END)
            self.analysis_results.insert(tk.END, error_msg)
    
    def plot_actual_vs_predicted(self, actual, predicted):
        """Plot actual vs predicted values"""
        try:
            # Clear previous plot
            self.ax.clear()
            
            # Create scatter plot
            self.ax.scatter(range(len(actual)), actual, color='blue', label='Actual')
            self.ax.scatter(range(len(predicted)), predicted, color='red', label='Predicted')
            self.ax.set_title('Actual vs Predicted Values')
            self.ax.set_xlabel('Sample Index')
            self.ax.set_ylabel('Value')
            self.ax.legend()
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot actual vs predicted: {str(e)}")
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        if not hasattr(self, 'rfc') or not hasattr(self.rfc, 'classes_'):
            messagebox.showerror("Error", "Please train the model first!")
            return
        
        try:
            # Evaluate Random Forest
            y_pred = self.rfc.predict(self.X_test_scaled)
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)
            
            # Update results
            self.analysis_results.delete(1.0, tk.END)
            self.analysis_results.insert(tk.END, "Model Evaluation Results\n\n")
            self.analysis_results.insert(tk.END, f"Accuracy: {accuracy:.4f}\n\n")
            self.analysis_results.insert(tk.END, "Classification Report:\n")
            self.analysis_results.insert(tk.END, report)
            
            # Plot confusion matrix
            self.plot_confusion_matrix(self.y_test, y_pred)
        except Exception as e:
            error_msg = f"Failed to evaluate model: {str(e)}\n\n{traceback.format_exc()}"
            messagebox.showerror("Error", error_msg)
            self.analysis_results.delete(1.0, tk.END)
            self.analysis_results.insert(tk.END, error_msg)
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        try:
            # Clear previous plot
            self.ax.clear()
            
            # Create confusion matrix
            cm = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=self.ax)
            self.ax.set_title('Confusion Matrix')
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot confusion matrix: {str(e)}")
    
    def analyze_soil_types(self):
        """Analyze soil types in the dataset"""
        if self.dataset is None:
            messagebox.showerror("Error", "Please upload a dataset first!")
            return
        
        try:
            # Check if Soil column exists
            if 'Soil' not in self.dataset.columns:
                messagebox.showinfo("Info", "No 'Soil' column found in dataset. Showing crop distribution instead.")
                # Plot crop distribution
                self.plot_crop_distribution()
                return
            
            # Count soil types
            soil_counts = self.dataset['Soil'].value_counts()
            
            # Update results
            self.analysis_results.delete(1.0, tk.END)
            self.analysis_results.insert(tk.END, "Soil Type Analysis\n\n")
            for soil, count in soil_counts.items():
                self.analysis_results.insert(tk.END, f"{soil}: {count} samples\n")
            
            # Plot soil distribution
            self.plot_soil_distribution(soil_counts)
        except Exception as e:
            error_msg = f"Failed to analyze soil types: {str(e)}\n\n{traceback.format_exc()}"
            messagebox.showerror("Error", error_msg)
            self.analysis_results.delete(1.0, tk.END)
            self.analysis_results.insert(tk.END, error_msg)
    
    def plot_soil_distribution(self, soil_counts):
        """Plot soil type distribution"""
        try:
            # Clear previous plot
            self.ax.clear()
            
            # Create pie chart
            self.ax.pie(soil_counts.values, labels=soil_counts.index, autopct='%1.1f%%', 
                       shadow=True, startangle=90)
            self.ax.set_title('Soil Type Distribution')
            self.ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot soil distribution: {str(e)}")
    
    def plot_crop_distribution(self):
        """Plot crop distribution"""
        try:
            # Clear previous plot
            self.ax.clear()
            
            # Count crop types
            crop_counts = self.dataset['label'].value_counts()
            
            # Create bar chart
            self.ax.bar(crop_counts.index, crop_counts.values)
            self.ax.set_title('Crop Distribution')
            self.ax.set_xlabel('Crop Type')
            self.ax.set_ylabel('Count')
            plt.xticks(rotation=90)
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot crop distribution: {str(e)}")
    
    def predict_crop(self):
        """Predict crop based on input parameters"""
        if not hasattr(self, 'rfc') or not hasattr(self.rfc, 'classes_'):
            messagebox.showerror("Error", "Please train the model first!")
            return
        
        try:
            # Get input values
            inputs = {}
            for key, var in self.input_vars.items():
                value = var.get().strip()
                if not value:
                    messagebox.showerror("Error", f"Please enter a value for {key}")
                    return
                try:
                    inputs[key] = float(value)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid value for {key}. Please enter a number.")
                    return
            
            # Create feature array
            features = np.array([[
                inputs['N'], inputs['P'], inputs['K'], 
                inputs['temperature'], inputs['humidity'], 
                inputs['pH'], inputs['rainfall']
            ]])
            
            # Transform features
            transformed_features = self.ms.transform(features)
            transformed_features = self.sc.transform(transformed_features)
            
            # Predict
            predicted_crop = self.rfc.predict(transformed_features)[0]
            
            # Get recommendations
            recommended_fertilizers = self.natural_fertilizers.get(predicted_crop, ["Unknown"])
            price_estimate = self.price.get(predicted_crop, ["Unknown"])
            
            # Calculate suitability score (simplified example)
            suitability_score = random.uniform(0.7, 1.0)  # In a real app, this would be calculated
            
            # Update results
            self.crop_result_label.config(text=f"Recommended Crop: {predicted_crop.upper()}")
            self.fertilizer_label.config(text=", ".join(recommended_fertilizers))
            self.price_label.config(text=", ".join(price_estimate))
            
            # Update suitability gauge
            self.update_suitability_gauge(suitability_score)
            
            # Add to history
            self.add_to_prediction_history(predicted_crop, inputs, suitability_score)
        except Exception as e:
            error_msg = f"Failed to predict crop: {str(e)}\n\n{traceback.format_exc()}"
            messagebox.showerror("Error", error_msg)
    
    def update_suitability_gauge(self, score):
        """Update the suitability gauge visualization"""
        try:
            # Clear canvas
            self.suitability_canvas.delete("all")
            
            # Draw gauge background
            width = self.suitability_canvas.winfo_width()
            if width < 10:  # Not yet rendered
                width = 300
            height = 30
            
            # Draw background
            self.suitability_canvas.create_rectangle(0, 0, width, height, fill="#f0f0f0", outline="")
            
            # Draw gauge
            gauge_width = int(width * score)
            
            # Color based on score
            if score >= 0.8:
                color = "#4CAF50"  # Green
            elif score >= 0.6:
                color = "#FFC107"  # Yellow
            else:
                color = "#F44336"  # Red
            
            self.suitability_canvas.create_rectangle(0, 0, gauge_width, height, fill=color, outline="")
            
            # Add score text
            self.suitability_canvas.create_text(width/2, height/2, text=f"{score:.2f}", fill="black", font=('Arial', 12, 'bold'))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update suitability gauge: {str(e)}")
    
    def add_to_prediction_history(self, crop, inputs, score):
        """Add prediction to history"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        history_entry = f"[{timestamp}] Crop: {crop}, Score: {score:.2f}\n"
        self.history_text.insert(tk.END, history_entry)
        self.history_text.see(tk.END)
    
    def clear_prediction_fields(self):
        """Clear all prediction input fields"""
        for var in self.input_vars.values():
            var.set("")
    
    def use_iot_data(self):
        """Use latest IoT data for prediction"""
        if not self.iot_simulation_running:
            messagebox.showinfo("Info", "IoT simulation is not running. Please start it first.")
            return
        
        try:
            # Get latest IoT data
            if not self.iot_data['nitrogen'] or not self.iot_data['phosphorus'] or not self.iot_data['potassium']:
                messagebox.showerror("Error", "Not enough IoT data available yet.")
                return
                
            self.input_vars['N'].set(str(self.iot_data['nitrogen'][-1]))
            self.input_vars['P'].set(str(self.iot_data['phosphorus'][-1]))
            self.input_vars['K'].set(str(self.iot_data['potassium'][-1]))
            self.input_vars['temperature'].set(str(self.iot_data['temperature'][-1]))
            self.input_vars['humidity'].set(str(self.iot_data['humidity'][-1]))
            self.input_vars['pH'].set(str(7.0))  # Assuming pH is not in IoT data
            self.input_vars['rainfall'].set(str(50.0))  # Assuming rainfall is not in IoT data
            
            messagebox.showinfo("Success", "IoT data loaded into prediction fields!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to use IoT data: {str(e)}")
    
    def toggle_iot_simulation(self):
        """Start or stop IoT simulation"""
        if self.iot_simulation_running:
            # Stop simulation
            self.iot_simulation_running = False
            self.iot_btn_text.set("Start IoT Simulation")
            self.status_indicator.itemconfig(1, fill='gray')
            self.status_label.config(text="IoT Status: Disconnected")
            
            # Wait for thread to finish
            if self.iot_thread and self.iot_thread.is_alive():
                self.iot_thread.join(timeout=1.0)
        else:
            # Start simulation
            self.iot_simulation_running = True
            self.iot_btn_text.set("Stop IoT Simulation")
            self.status_indicator.itemconfig(1, fill='green')
            self.status_label.config(text="IoT Status: Connected")
            
            # Start simulation thread
            self.iot_thread = threading.Thread(target=self.run_iot_simulation, daemon=True)
            self.iot_thread.start()
    
    def run_iot_simulation(self):
        """Run IoT data simulation"""
        while self.iot_simulation_running:
            try:
                # Generate random data
                temp = random.uniform(20.0, 35.0)
                humidity = random.uniform(30.0, 90.0)
                soil_moisture = random.uniform(20.0, 80.0)
                nitrogen = random.uniform(30.0, 150.0)
                phosphorus = random.uniform(20.0, 100.0)
                potassium = random.uniform(20.0, 120.0)
                
                # Add to data store
                self.iot_data['temperature'].append(temp)
                self.iot_data['humidity'].append(humidity)
                self.iot_data['soil_moisture'].append(soil_moisture)
                self.iot_data['nitrogen'].append(nitrogen)
                self.iot_data['phosphorus'].append(phosphorus)
                self.iot_data['potassium'].append(potassium)
                
                # Keep only last 50 readings
                for key in self.iot_data:
                    if len(self.iot_data[key]) > 50:
                        self.iot_data[key] = self.iot_data[key][-50:]
                
                # Update UI
                self.root.after(0, self.update_iot_readings, temp, humidity, soil_moisture, nitrogen, phosphorus, potassium)
                self.root.after(0, self.update_iot_graph)
                
                # Wait for next sample
                try:
                    sample_rate = float(self.sample_rate.get())
                except ValueError:
                    sample_rate = 5.0
                time.sleep(sample_rate)
            except Exception as e:
                print(f"IoT simulation error: {str(e)}")
                time.sleep(5)
    
    def update_iot_readings(self, temp, humidity, soil_moisture, nitrogen, phosphorus, potassium):
        """Update IoT readings display"""
        try:
            self.reading_labels['temperature'].config(text=f"{temp:.1f}")
            self.reading_labels['humidity'].config(text=f"{humidity:.1f}")
            self.reading_labels['soil_moisture'].config(text=f"{soil_moisture:.1f}")
            self.reading_labels['nitrogen'].config(text=f"{nitrogen:.1f}")
            self.reading_labels['phosphorus'].config(text=f"{phosphorus:.1f}")
            self.reading_labels['potassium'].config(text=f"{potassium:.1f}")
        except Exception as e:
            print(f"Error updating IoT readings: {str(e)}")
    
    def update_iot_graph(self, event=None):
        """Update IoT data graph"""
        try:
            if not self.iot_data or not self.iot_data[self.selected_param.get()]:
                return
            
            # Clear previous plot
            self.iot_ax.clear()
            
            # Get selected parameter data
            param = self.selected_param.get()
            data = self.iot_data[param]
            
            # Plot data
            self.iot_ax.plot(range(len(data)), data, 'b-')
            self.iot_ax.set_title(f'{param.capitalize()} Over Time')
            self.iot_ax.set_xlabel('Sample')
            self.iot_ax.set_ylabel(param.capitalize())
            self.iot_fig.tight_layout()
            self.iot_canvas.draw()
        except Exception as e:
            print(f"Error updating IoT graph: {str(e)}")
    
    def export_iot_data(self):
        """Export IoT data to CSV"""
        if not self.iot_data or not self.iot_data['temperature']:
            messagebox.showinfo("Info", "No IoT data to export.")
            return
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.iot_data)
            
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"IoT data exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")
    
    def analyze_micronutrients(self):
        """Analyze micronutrient levels"""
        try:
            # Get input values
            micronutrient_levels = {}
            for key, var in self.micronutrient_vars.items():
                value = var.get().strip()
                if not value:
                    messagebox.showerror("Error", f"Please enter a value for {key}")
                    return
                try:
                    micronutrient_levels[key] = float(value)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid value for {key}. Please enter a number.")
                    return
            
            # Analyze levels
            analysis_results = {}
            recommendations = []
            
            for nutrient, level in micronutrient_levels.items():
                thresholds = self.micronutrient_thresholds[nutrient]
                
                if level < thresholds['low']:
                    status = "Deficient"
                    recommendations.append(f"Increase {nutrient} levels with appropriate fertilizers.")
                elif level > thresholds['high']:
                    status = "Excessive"
                    recommendations.append(f"Reduce {nutrient} application to avoid toxicity.")
                elif level >= thresholds['low'] and level < thresholds['optimal']:
                    status = "Low"
                    recommendations.append(f"Slightly increase {nutrient} for optimal growth.")
                else:
                    status = "Optimal"
                
                analysis_results[nutrient] = {
                    'level': level,
                    'status': status
                }
            
            # Update soil info
            selected_soil = self.soil_type.get()
            soil_info = self.soil_types.get(selected_soil, {})
            
            self.soil_info_text.delete(1.0, tk.END)
            if soil_info:
                self.soil_info_text.insert(tk.END, f"Soil Type: {selected_soil}\n\n")
                self.soil_info_text.insert(tk.END, f"Description: {soil_info['description']}\n")
                self.soil_info_text.insert(tk.END, f"Drainage: {soil_info['drainage']}\n")
                self.soil_info_text.insert(tk.END, f"Water Retention: {soil_info['water_retention']}\n")
                self.soil_info_text.insert(tk.END, f"Suitable Crops: {', '.join(soil_info['suitable_crops'])}\n")
            
            # Update recommendations
            self.rec_text.delete(1.0, tk.END)
            for rec in recommendations:
                self.rec_text.insert(tk.END, f"• {rec}\n")
            
            # Plot micronutrient levels
            self.plot_micronutrient_levels(micronutrient_levels, analysis_results)
        except Exception as e:
            error_msg = f"Failed to analyze micronutrients: {str(e)}\n\n{traceback.format_exc()}"
            messagebox.showerror("Error", error_msg)
    
    def plot_micronutrient_levels(self, levels, analysis):
        """Plot micronutrient levels with status indicators"""
        try:
            # Clear previous plot
            self.micro_ax.clear()
            
            # Prepare data
            nutrients = list(levels.keys())
            values = list(levels.values())
            
            # Create color map based on status
            colors = []
            for nutrient in nutrients:
                status = analysis[nutrient]['status']
                if status == "Deficient":
                    colors.append('red')
                elif status == "Low":
                    colors.append('orange')
                elif status == "Optimal":
                    colors.append('green')
                else:  # Excessive
                    colors.append('purple')
            
            # Create bar chart
            bars = self.micro_ax.bar(nutrients, values, color=colors)
            
            # Add threshold lines
            for i, nutrient in enumerate(nutrients):
                thresholds = self.micronutrient_thresholds[nutrient]
                self.micro_ax.hlines(y=thresholds['low'], xmin=i-0.4, xmax=i+0.4, colors='red', linestyles='dashed')
                self.micro_ax.hlines(y=thresholds['optimal'], xmin=i-0.4, xmax=i+0.4, colors='green', linestyles='dashed')
                self.micro_ax.hlines(y=thresholds['high'], xmin=i-0.4, xmax=i+0.4, colors='blue', linestyles='dashed')
            
            # Add labels and title
            self.micro_ax.set_title('Micronutrient Analysis')
            self.micro_ax.set_xlabel('Micronutrients')
            self.micro_ax.set_ylabel('Level (mg/kg)')
            
            # Add status labels above bars
            for bar, nutrient in zip(bars, nutrients):
                status = analysis[nutrient]['status']
                self.micro_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                  status, ha='center', va='bottom', rotation=0, fontsize=8)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label='Deficient'),
                Patch(facecolor='orange', label='Low'),
                Patch(facecolor='green', label='Optimal'),
                Patch(facecolor='purple', label='Excessive')
            ]
            self.micro_ax.legend(handles=legend_elements, loc='upper right')
            
            self.micro_fig.tight_layout()
            self.micro_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot micronutrient levels: {str(e)}")
    
    def generate_random_micronutrients(self):
        """Generate random micronutrient data for demonstration"""
        for nutrient in self.micronutrient_vars:
            thresholds = self.micronutrient_thresholds[nutrient]
            # Generate a value that could be in any range (deficient, low, optimal, or excessive)
            max_val = thresholds['high'] * 1.5
            value = random.uniform(0, max_val)
            self.micronutrient_vars[nutrient].set(f"{value:.2f}")
    
    def export_results(self):
        """Export analysis results"""
        if not hasattr(self, 'analysis_results') or not self.analysis_results.get(1.0, tk.END).strip():
            messagebox.showinfo("Info", "No analysis results to export.")
            return
        
        try:
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write("Soil Analysis & Micronutrient Classification Results\n")
                    f.write("="*50 + "\n\n")
                    f.write(self.analysis_results.get(1.0, tk.END))
                    
                    # If we have micronutrient data, add that too
                    if hasattr(self, 'rec_text') and self.rec_text.get(1.0, tk.END).strip():
                        f.write("\n\nMicronutrient Recommendations:\n")
                        f.write("-"*50 + "\n")
                        f.write(self.rec_text.get(1.0, tk.END))
                    
                messagebox.showinfo("Success", f"Results exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")

# Create and run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = SoilAnalysisApp(root)
    root.mainloop()