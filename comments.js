// Create web server

// Dependencies
const express = require('express');
const router = express.Router();
const { getComments, addComment, deleteComment } = require('../controllers/comments');

// Routes
router
  .route('/')
  .get(getComments)
  .post(addComment);

router
  .route('/:id')
  .delete(deleteComment);

// Export module
module.exports = router;