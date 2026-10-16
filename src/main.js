import { createStandaloneApplication } from './standalone/application.js';
import { describeError } from './standalone/errors.js';

const application = createStandaloneApplication({
  googleApiKey: import.meta.env.GOOGLE_MAPS_API_KEY,
  cesiumToken: import.meta.env.CESIUM_ION_TOKEN,
  allowQaRegistration: import.meta.env.DEV,
});

application.start().then((components) => {
  window.CIVICPULSE_APP = {
    application,
    components,
    viewer: components.scene?.viewer,
    scene: components.scene,
    data: components.data,
    dataManager: components.data?.dataManager,
    controls: components.controls,
  };
  window.dispatchEvent(new CustomEvent('civicpulse:ready', { detail: window.CIVICPULSE_APP }));
  return import('./civicpulse/civicpulse-intel.js');
}).catch((error) => {
  console.error("God's Eye View initialization failed:", error);
  const loaderStatus = document.querySelector('#loading-screen .loader-status');
  if (loaderStatus) {
    loaderStatus.textContent = `Error: ${describeError(error)}`;
    loaderStatus.style.color = '#ff4444';
  }
});

export { application };
