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

// Load metrics on page load
document.addEventListener("DOMContentLoaded", async function () {
  const metricsContent = document.getElementById("metrics-content");

  try {
    const response = await fetch("/task3/metrics");
    const metrics = await response.json();

    let html = "<ul>";
    for (const [key, value] of Object.entries(metrics)) {
      html += `<li><strong>${key}:</strong> ${value}</li>`;
    }
    html += "</ul>";
    metricsContent.innerHTML = html;
  } catch (error) {
    metricsContent.innerHTML = "<p>Error loading metrics.</p>";
  }
});
