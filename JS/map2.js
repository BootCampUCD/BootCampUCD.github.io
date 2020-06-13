map2;


var mapboxAccessToken = {API_KEY};
var map = L.map('map').setView([37.8, -96], 4);

L.tileLayer('https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}?access_token=' + mapboxAccessToken, {
    id: 'mapbox/light-v9',
    // attribution: ,
    tileSize: 512,
    zoomOffset: -1
    access_token: API_KEY,
}).addTo(map);

L.geoJson(statesData).addTo(map);


