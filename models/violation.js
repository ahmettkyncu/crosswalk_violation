const mongoose = require('mongoose');

const violationSchema = new mongoose.Schema({
    plateText: {
        type: String,
        default: "OKUNAMADI",
        index: true
    },
    imageFilename: {
        type: String,
        required: true
    },
    reason: {
        type: String,
        required: true
    },
    cameraName: {
        type: String,
        required: true,
        index: true
    },
    trackId: {  
        type: String,
        required: true,
        index: true
    },
    status: {   
        type: String,
        enum: ['violation', 'normal'],
        default: 'violation'
    },
    timestamp: {
        type: Date,
        default: Date.now,
        index: true
    }
});

module.exports = mongoose.model('Violation', violationSchema);