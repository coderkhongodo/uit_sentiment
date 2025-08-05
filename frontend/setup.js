#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🚀 Setting up React Frontend...\n');

// Check if Node.js and npm are installed
try {
    execSync('node --version', { stdio: 'ignore' });
    execSync('npm --version', { stdio: 'ignore' });
} catch (error) {
    console.error('❌ Node.js and npm are required. Please install them first.');
    process.exit(1);
}

// Install dependencies
console.log('📦 Installing dependencies...');
try {
    execSync('npm install', { stdio: 'inherit' });
    console.log('✅ Dependencies installed successfully!\n');
} catch (error) {
    console.error('❌ Failed to install dependencies');
    process.exit(1);
}

// Create .env file if it doesn't exist
const envPath = path.join(__dirname, '.env');
if (!fs.existsSync(envPath)) {
    const envContent = `# React App Environment Variables
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000/ws
GENERATE_SOURCEMAP=false
`;

    fs.writeFileSync(envPath, envContent);
    console.log('✅ Created .env file');
}

// Create additional directories if needed
const directories = [
    'src/components',
    'src/pages',
    'src/hooks',
    'src/services',
    'src/context',
    'src/utils'
];

directories.forEach(dir => {
    const dirPath = path.join(__dirname, dir);
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
        console.log(`✅ Created directory: ${dir}`);
    }
});

console.log('\n🎉 Frontend setup completed!');
console.log('\nTo start the development server:');
console.log('  npm start');
console.log('\nThe app will be available at: http://localhost:3000');
console.log('Make sure the backend is running at: http://localhost:8000');