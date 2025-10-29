document.getElementById("predictForm").addEventListener("submit", async function (e) {
  e.preventDefault();
  const formData = new FormData(this);
  
  const response = await fetch("/task3/predict", {
    method: "POST",
    body: formData,
  });
  
  const data = await response.json();
  const resultBox = document.getElementById("result-box");
  const predictionText = document.getElementById("prediction-text");
  
  predictionText.textContent = data.prediction;
  resultBox.classList.remove("hidden");
});
