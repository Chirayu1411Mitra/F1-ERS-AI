document.getElementById("runBtn").addEventListener("click", async () => {
  const trackCondition = document.getElementById("trackCondition").value;
  const drsEnabled = document.getElementById("drsToggle").checked;
  const loader = document.getElementById("loader");
  const results = document.getElementById("results");

  loader.classList.remove("hidden");
  results.classList.add("hidden");

  try {
    const response = await fetch("/run-ai", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        track_condition: trackCondition,
        drs_enabled: drsEnabled,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    loader.classList.add("hidden");
    results.classList.remove("hidden");

    document.getElementById("lapTime").textContent = data.lap_time;
    document.getElementById("strategyOutput").textContent =
      data.strategy.join(", ");
    document.getElementById("lapImage").src =
      data.image_path + "?t=" + new Date().getTime();
  } catch (error) {
    console.error("Error:", error);
    loader.classList.add("hidden");
    alert(
      "An error occurred while running the optimization. Please try again."
    );
  }
});
