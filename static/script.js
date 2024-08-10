function validateForm() {
  const rank = document.getElementById('rank').value;
  const rankError = document.getElementById('rankError');

  if (!rank || isNaN(rank)) {
    rankError.style.display = 'block';
    return false;
  } else {
    rankError.style.display = 'none';
    return true;
  }
}
