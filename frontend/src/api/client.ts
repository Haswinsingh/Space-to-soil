import axios from 'axios';

export const api = axios.create({
  baseURL: '/api',
  timeout: 600000,
});

export default api;
