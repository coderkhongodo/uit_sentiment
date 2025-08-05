
# Vietnamese Sentiment Analysis Web Application Startup Script
echo "🚀 Starting Vietnamese Sentiment Analysis Web Application..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if model files exist
check_model_files() {
    if [ -d "saved_results/final_model" ] && [ -f "saved_results/final_model/model.safetensors" ]; then
        print_success "PhoBERT model files found"
        return 0
    else
        print_warning "PhoBERT model files not found in saved_results/final_model/"
        print_warning "Please ensure the trained model is available before starting"
        return 1
    fi
}

# Start manually (development mode)
start_manual() {
    print_status "Starting application manually..."
    
    # Check Python and Node.js
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        return 1
    fi
    
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed"
        return 1
    fi
    
    # Start backend
    print_status "Starting backend..."
    cd backend
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install -r requirements.txt
    
    # Start backend in background
    python run.py &
    BACKEND_PID=$!
    cd ..
    
    # Wait a bit for backend to start
    sleep 5
    
    # Start frontend
    print_status "Starting frontend..."
    cd frontend
    
    if [ ! -d "node_modules" ]; then
        print_status "Installing frontend dependencies..."
        npm install
    fi
    
    # Start frontend in background
    npm start &
    FRONTEND_PID=$!
    cd ..
    
    print_success "Application started successfully!"
    echo ""
    echo "🌐 Application URLs:"
    echo "   Frontend: http://localhost:3000"
    echo "   Backend API: http://localhost:8000"
    echo "   API Documentation: http://localhost:8000/docs"
    echo ""
    echo "🛑 To stop the application:"
    echo "   Press Ctrl+C or run: kill $BACKEND_PID $FRONTEND_PID"
    
    # Wait for user to stop
    wait
}

# Main execution
main() {
    echo "🇻🇳 Vietnamese Sentiment Analysis - Local Development"
    echo "====================================================="
    echo ""
    
    # Check model files
    check_model_files
    
    # Ask user for startup method
    echo "Choose startup method:"
    echo "1) Manual (Development)"
    echo "2) Exit"
    echo ""
    read -p "Enter your choice (1-2): " choice
    
    case $choice in
        1)
            start_manual
            ;;
        2)
            print_status "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
}

# Run main function
main