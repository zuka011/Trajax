/**
 * Lazy loading for iframe visualizations.
 * - Iframes only load when they enter the viewport
 * - Iframes go "idle" (src removed) when they scroll out of view
 */
document$.subscribe(({ body }) => {
  const iframes = body.querySelectorAll('iframe[src*="visualizations/"]');
  
  if (!iframes.length) return;
  
  const observerOptions = {
    root: null,
    rootMargin: '100px',
    threshold: 0.1
  };
  
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      const iframe = entry.target;
      const originalSrc = iframe.dataset.src;
      
      if (entry.isIntersecting) {
        if (originalSrc && iframe.src !== originalSrc) {
          iframe.src = originalSrc;
        }
      } else {
        if (iframe.src && iframe.src !== 'about:blank') {
          iframe.dataset.src = iframe.src;
          iframe.src = 'about:blank';
        }
      }
    });
  }, observerOptions);
  
  iframes.forEach(iframe => {
    iframe.dataset.src = iframe.src;
    iframe.src = 'about:blank';
    iframe.setAttribute('loading', 'lazy');
    observer.observe(iframe);
  });
});
