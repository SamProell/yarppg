import pyqtgraph


def plain_image_item(data):
    """Create a `pyqtgraph.ImageView` showing only the actual image."""
    img_item = pyqtgraph.image(data)
    img_item.ui.histogram.hide()
    img_item.ui.roiBtn.hide()
    img_item.ui.menuBtn.hide()
    return img_item
