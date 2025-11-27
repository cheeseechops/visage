# People Viewer - Web App UI Mockup

A clean, responsive web interface for browsing and managing thousands of cropped people images taken from screenshots (e.g., Instagram posts). The app provides an intuitive way to browse, tag, and confirm people identified through facial recognition.

## Features

- **All People Page**: Grid view of all detected people with thumbnails and image counts
- **Person Gallery**: Individual person pages showing all their images
- **All Images Page**: Browse all cropped images with advanced filtering
- **Image Detail View**: Detailed view with original screenshot and facial recognition suggestions
- **Manual Tagging**: Interface for assigning unidentified people to known persons
- **Import System**: Simple folder upload to 'imported' directory
- **Auto-Cropper**: YOLO-based person detection and cropping from imported images
- **Favorites System**: Mark people and images as favorites
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Tech Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, Tailwind CSS, Vanilla JavaScript
- **Styling**: Modern, clean UI with hover effects and smooth transitions

## Installation & Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python app.py
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Pages Overview

### 1. Home Page ("All People")
- Grid of person cards showing thumbnails, names, and image counts
- Quick favorite toggling
- Status badges for unidentified people
- Statistics dashboard

### 2. Person Gallery Page
- Header with person details and favorite toggle
- Grid of all images for that person
- Quick access to image details
- Tag display and management

### 3. All Images Page
- Comprehensive filter bar (favorites, person, search)
- Grid view of all cropped images
- Person badges and tag indicators
- Quick navigation to person galleries

### 4. Image Detail Page
- Side-by-side view of cropped person and original screenshot
- Facial recognition suggestions with confidence scores
- Confirmation/rejection buttons for unidentified people
- Tag management and image actions

### 5. Manual Tagging Page
- Interface for assigning unidentified images
- Quick assign buttons based on AI suggestions
- Create new person functionality
- Skip option for difficult cases

### 6. Import Page
- Simple folder selection interface
- Drag and drop support for image folders
- Automatic upload to 'imported' folder
- Preview of recently imported images

### 7. Auto-Cropper Page
- Process imported images with YOLO person detection
- Click to select images from imported folder
- Automatic cropping and saving to 'cropped' folder
- Database integration for cropped images
- Preview of recently cropped results

## Mock Data

The application includes realistic mock data to demonstrate all features:
- 6 people (3 confirmed, 3 unidentified)
- 3 sample images with facial recognition suggestions
- Various tags and metadata
- Favorite status examples

## API Endpoints

The app includes RESTful API endpoints for future backend integration:
- `GET /api/people` - Get all people data
- `GET /api/images` - Get all images data
- `POST /api/toggle_favorite/<person_id>` - Toggle person favorite status
- `POST /api/toggle_image_favorite/<image_id>` - Toggle image favorite status

## Customization

### Adding Real Data
Replace the mock data in `app.py` with your actual database queries:
- `MOCK_PEOPLE` - Your people table data
- `MOCK_IMAGES` - Your images table data

### Styling
The app uses Tailwind CSS for styling. Customize colors and components in the base template:
- Primary color: `#4F46E5` (indigo)
- Custom animations and hover effects
- Responsive grid layouts

### Database Integration
The app is structured to easily integrate with SQLite or any other database:
- Flask-SQLAlchemy for ORM
- Database models for People and Images
- Migration scripts for schema management

## Future Enhancements

- **Real-time Updates**: WebSocket integration for live updates
- **Bulk Operations**: Select multiple images for batch tagging
- **Advanced Search**: Full-text search with filters
- **Export Features**: Export tagged data to various formats
- **User Authentication**: Multi-user support with permissions
- **Image Upload**: Direct upload and processing interface

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

This project is provided as a UI mockup for demonstration purposes. 