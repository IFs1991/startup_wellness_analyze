import React, { useState } from 'react';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography';
import NoteService from '../services/NoteService'; // NoteServiceをインポート

const NoteInput = ({ analysisId }) => {
  const [noteText, setNoteText] = useState('');

  const handleNoteTextChange = (event) => {
    setNoteText(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    try {
      await NoteService.createNote({ analysisId, content: noteText }); // APIコールでメモを保存
      setNoteText(''); // 入力欄をクリア
      // 保存成功のメッセージなどを表示
    } catch (error) {
      // エラー処理
    }
  };

  return (
    <Grid container spacing={2} justifyContent="center">
      <Grid item xs={12}>
        <Typography variant="h6" align="center" gutterBottom>
          Add Note
        </Typography>
      </Grid>
      <Grid item xs={12} md={8}>
        <TextField
          label="Your Note"
          fullWidth
          multiline
          rows={4}
          value={noteText}
          onChange={handleNoteTextChange}
        />
      </Grid>
      <Grid item xs={12} align="center">
        <Button type="submit" variant="contained" color="primary" onClick={handleSubmit}>
          Save Note
        </Button>
      </Grid>
    </Grid>
  );
};

export default NoteInput;