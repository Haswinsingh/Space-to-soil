import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_crop_report(
    output_path: str,
    image_path: str,
    processed_image_path: str,
    crop_health_class: str,
    confidence: float,
    predicted_yield: float,
    classical_results: dict,
    quantum_results: dict,
    recommendations: list
):
    """
    Generates a professional agricultural health and yield analysis PDF report.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0d1b2a")
    c_secondary = colors.HexColor("#00ffd5")
    c_text = colors.HexColor("#1e293b")
    c_accent = colors.HexColor("#10b981")
    
    # Custom styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=colors.white,
        spaceAfter=10,
        alignment=1 # Center
    )
    
    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        textColor=colors.HexColor("#94a3b8"),
        spaceAfter=15,
        alignment=1
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=c_primary,
        spaceBefore=15,
        spaceAfter=8,
        borderPadding=2
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=c_text,
        leading=14
    )
    
    bold_body = ParagraphStyle(
        'ReportBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    table_text = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=c_text
    )

    story = []
    
    # --- HEADER BANNER ---
    header_data = [
        [Paragraph("QUANTUMCROP AI – ANALYSIS REPORT", title_style)],
        [Paragraph("Quantum Remote Sensing & Yield Prediction Platform", subtitle_style)]
    ]
    header_table = Table(header_data, colWidths=[doc.width])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_primary),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 15))
    
    # --- METADATA SECTION ---
    meta_data = [
        [Paragraph("<b>Date:</b> 2026-07-26", body_style), Paragraph("<b>Sensor:</b> Simulated Sentinel-2 Multispectral", body_style)],
        [Paragraph("<b>Analysis Engine:</b> Classical + QML (Hybrid)", body_style), Paragraph("<b>Location ID:</b> Field-Alpha-32", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[doc.width/2.0, doc.width/2.0])
    meta_table.setStyle(TableStyle([
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))
    
    # --- IMAGERY FLOW ---
    story.append(Paragraph("Satellite & Index Imagery", h1_style))
    img_data = []
    
    # Load and scale original and processed images if they exist
    col_width = doc.width / 2.0 - 10
    row_images = []
    
    if image_path and os.path.exists(image_path):
        try:
            row_images.append(Image(image_path, width=col_width, height=col_width * 0.75))
        except Exception:
            row_images.append(Paragraph("[Original Image Render Failed]", body_style))
    else:
        row_images.append(Paragraph("[Original Image Not Uploaded]", body_style))
        
    if processed_image_path and os.path.exists(processed_image_path):
        try:
            row_images.append(Image(processed_image_path, width=col_width, height=col_width * 0.75))
        except Exception:
            row_images.append(Paragraph("[NDVI Processed Render Failed]", body_style))
    else:
        row_images.append(Paragraph("[NDVI Index Map Not Generated]", body_style))
        
    img_data.append(row_images)
    img_data.append([Paragraph("<b>Uploaded Imagery (RGB)</b>", body_style), Paragraph("<b>Computed NDVI Index Map</b>", body_style)])
    
    img_table = Table(img_data, colWidths=[doc.width/2.0, doc.width/2.0])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 1), (-1, 1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 10),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 15))
    
    # --- PREDICTIONS SUMMARY ---
    story.append(Paragraph("Crop Health & Yield Prediction Summary", h1_style))
    
    summary_data = [
        [Paragraph("<b>Crop Health Status:</b>", body_style), Paragraph(f"<font color='{c_accent.hexval()}'><b>{crop_health_class}</b></font>", bold_body)],
        [Paragraph("<b>Prediction Confidence:</b>", body_style), Paragraph(f"{confidence * 100:.2f}%", body_style)],
        [Paragraph("<b>Estimated Crop Yield:</b>", body_style), Paragraph(f"<b>{predicted_yield:.2f} Tons / Hectare</b>", bold_body)]
    ]
    summary_table = Table(summary_data, colWidths=[150, doc.width - 150])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))
    
    # --- CLASSICAL VS QUANTUM COMPARISON ---
    story.append(Paragraph("Machine Learning Benchmarking (Classical vs Quantum)", h1_style))
    
    comp_headers = [
        Paragraph("<b>Model Name</b>", table_text),
        Paragraph("<b>Type</b>", table_text),
        Paragraph("<b>Health Classification</b>", table_text),
        Paragraph("<b>Confidence</b>", table_text),
        Paragraph("<b>Accuracy (Benchmark)</b>", table_text),
        Paragraph("<b>Inference Latency</b>", table_text)
    ]
    
    comp_rows = [comp_headers]
    
    # Add classical entries
    for name, metrics in classical_results.items():
        comp_rows.append([
            Paragraph(name.upper().replace("_", " "), table_text),
            Paragraph("Classical", table_text),
            Paragraph(metrics.get("class_name", crop_health_class), table_text),
            Paragraph(f"{metrics.get('confidence', 0.0)*100:.1f}%", table_text),
            Paragraph(f"{metrics.get('benchmark_acc', 0.88)*100:.1f}%", table_text),
            Paragraph("~0.002s", table_text)
        ])
        
    # Add quantum entries
    for name, metrics in quantum_results.items():
        comp_rows.append([
            Paragraph(name.upper(), table_text),
            Paragraph("Quantum (QSVM)", table_text),
            Paragraph(metrics.get("class_name", crop_health_class), table_text),
            Paragraph(f"{metrics.get('confidence', 0.0)*100:.1f}%", table_text),
            Paragraph(f"{metrics.get('benchmark_acc', 0.84)*100:.1f}%", table_text),
            Paragraph("~0.005s", table_text)
        ])
        
    comp_table = Table(comp_rows, colWidths=[100, 80, 110, 70, 90, 80])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#cbd5e1")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 15))
    
    # --- RECOMMENDATION ENGINE ---
    story.append(Paragraph("AI Assistant Recommendations", h1_style))
    
    rec_story = []
    for rec in recommendations:
        if isinstance(rec, dict):
            severity = rec.get("severity", "medium").upper()
            title = rec.get("title", "")
            msg = rec.get("message", "")
            rec_story.append(Paragraph(f"• <b>[{severity}] {title}</b>: {msg}", body_style))
        else:
            rec_story.append(Paragraph(f"• {rec}", body_style))
        rec_story.append(Spacer(1, 4))
        
    story.append(KeepTogether(rec_story))
    
    # Build Document
    doc.build(story)
