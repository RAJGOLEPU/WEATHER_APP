document.addEventListener('DOMContentLoaded', () => {
    const weatherDescription = document.getElementById('weatherDescription');
    const homeBtn = document.getElementById('homeBtn');
    const locationSelect = document.getElementById('locationSelect');

    // Function to fetch weather from Flask
    const fetchWeather = async (location) => {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ location: location }),
        });
        const data = await response.json();
        return data.prediction;
    };

    // Handle dropdown change
    locationSelect.addEventListener('change', async (event) => {
        const location = event.target.value;
        weatherDescription.textContent = 'Fetching weather...';
        const weatherData = await fetchWeather(location);
        weatherDescription.textContent = weatherData;
    });

    // Handle home button click
    homeBtn.addEventListener('click', () => {
        locationSelect.value = 'new-york';
        weatherDescription.textContent = 'Fetching weather...';
        fetchWeather('new-york').then(weatherData => {
            weatherDescription.textContent = weatherData;
        });
    });

    // Initial load
    locationSelect.dispatchEvent(new Event('change'));
});
