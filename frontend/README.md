# STT Control Panel Frontend

## Overview

This is the web-based control panel for the Adaptive Self-Learning Agentic AI System. It provides a comprehensive interface for managing all aspects of the STT system.

## Features

### 📊 Dashboard
- Real-time system health monitoring
- Agent statistics and performance metrics
- Data management statistics
- Model information display

### 🎤 Transcription
- Upload and transcribe audio files
- Choose between baseline or agent mode
- Auto-correction and error recording
- Real-time results display

### 💾 Data Management
- Browse and search failed transcription cases
- View case details and add corrections
- Prepare fine-tuning datasets
- Manage dataset versions

### 🔧 Fine-Tuning
- Monitor orchestrator status
- Trigger fine-tuning jobs
- View job history and status
- Force trigger options

### 🧊 Models
- View current model information
- See deployed model details
- Browse model version history
- Track model deployments

### 📈 Monitoring
- Performance metrics tracking
- Trend visualization
- WER/CER monitoring
- Historical data analysis

## Technology Stack

- **HTML5**: Structure and semantics
- **CSS3**: Modern styling with CSS variables
- **Vanilla JavaScript**: No framework dependencies
- **Font Awesome**: Icon library
- **REST API**: Backend communication

## File Structure

```
frontend/
├── index.html      # Main HTML structure
├── styles.css      # Complete styling
├── app.js          # Application logic and API integration
└── README.md       # This file
```

## Setup

1. Ensure the backend API is running:
   ```bash
   uvicorn src.control_panel_api:app --reload --port 8000
   ```

2. Access the frontend:
   ```
   http://localhost:8000/app
   ```

## API Integration

The frontend communicates with the backend API at `http://localhost:8000`. All API calls are handled in `app.js`.

### Key Functions

- `checkSystemHealth()`: Monitor system status
- `loadDashboard()`: Load dashboard data
- `transcribeAudio()`: Process audio files
- `loadFailedCases()`: Retrieve failed cases
- `prepareDataset()`: Create fine-tuning datasets
- `triggerFinetuning()`: Start fine-tuning jobs

## Customization

### Change API URL

Edit `app.js`:
```javascript
const API_BASE_URL = 'http://your-api-url:port';
```

### Modify Styling

Edit `styles.css` to customize:
- Colors (CSS variables in `:root`)
- Layout and spacing
- Component styles
- Responsive breakpoints

### Add Features

1. Add HTML structure in `index.html`
2. Add styles in `styles.css`
3. Add logic in `app.js`
4. Ensure backend API endpoints exist

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (responsive design)

## Development

### Enable Debug Mode

Open browser console (F12) to see:
- API request/response logs
- Error messages
- State changes

### Test API Endpoints

Use the interactive API docs:
```
http://localhost:8000/docs
```

## Future Enhancements

- [ ] Chart.js integration for visual trends
- [ ] Real-time WebSocket updates
- [ ] User authentication
- [ ] Dark mode toggle
- [ ] Batch audio upload
- [ ] Export data functionality
- [ ] Advanced filtering options
- [ ] Custom dashboard widgets

## Troubleshooting

### Frontend Not Loading

1. Check API is running: `curl http://localhost:8000/api/health`
2. Clear browser cache
3. Check browser console for errors
4. Verify frontend files are in correct location

### API Connection Failed

1. Verify API URL in `app.js`
2. Check CORS settings in backend
3. Ensure no firewall blocking requests
4. Test API with curl or Postman

## License

Part of the Adaptive Self-Learning Agentic AI System project.

